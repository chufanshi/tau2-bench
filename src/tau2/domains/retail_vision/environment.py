"""Environment and task loaders for Track A retail visual evidence."""

import base64
import hashlib
from pathlib import Path
from typing import Optional

from tau2.data_model.tasks import Task
from tau2.domains.retail.utils import RETAIL_POLICY_PATH
from tau2.domains.retail_vision.data_model import RetailVisionDB
from tau2.domains.retail_vision.tools import RetailVisionTools
from tau2.domains.retail_vision.utils import (
    RETAIL_DB_PATH,
    RETAIL_VISION_POLICY_PATH,
    RETAIL_VISION_TASK_SET_PATH,
)
from tau2.environment.environment import Environment
from tau2.utils import load_file


def get_environment(
    db: Optional[RetailVisionDB] = None,
    solo_mode: bool = False,
) -> Environment:
    """Build the retail environment with the visual-policy action extension."""
    if solo_mode:
        raise ValueError("Retail-vision does not support solo mode")
    if db is None:
        db = RetailVisionDB.load(RETAIL_DB_PATH)
    policy = Path(RETAIL_POLICY_PATH).read_text()
    visual_policy = Path(RETAIL_VISION_POLICY_PATH).read_text()
    return Environment(
        domain_name="retail-vision",
        policy=f"{policy.rstrip()}\n\n{visual_policy.lstrip()}",
        tools=RetailVisionTools(db),
    )


def get_tasks(task_split_name: Optional[str] = "base") -> list[Task]:
    """Load grounded retail-vision tasks."""
    tasks = [Task.model_validate(task) for task in load_file(RETAIL_VISION_TASK_SET_PATH)]
    if task_split_name is None:
        return tasks
    splits = get_tasks_split()
    if task_split_name not in splits:
        raise ValueError(
            f"Invalid task split name: {task_split_name}. Valid splits: {splits.keys()}"
        )
    selected = set(splits[task_split_name])
    return [task for task in tasks if task.id in selected]


def get_tasks_split() -> dict[str, list[str]]:
    """Load task IDs for each retail-vision split."""
    split_path = (
        Path(RETAIL_VISION_TASK_SET_PATH).parent
        / f"split_{Path(RETAIL_VISION_TASK_SET_PATH).stem}.json"
    )
    return load_file(split_path)


def get_released_tasks(
    release_dir: Path,
    *,
    task_split_name: Optional[str] = "base",
    task_ids: Optional[list[str]] = None,
    num_tasks: Optional[int] = None,
) -> list[Task]:
    """Load and hash-validate agent-safe tasks/assets from a published release."""

    release_dir = release_dir.expanduser().resolve()
    raw_tasks = load_file(release_dir / "tasks.json")
    registry = load_file(release_dir / "image_registry.json")
    if not isinstance(raw_tasks, list):
        raise ValueError("release tasks.json must be a list")
    if registry.get("schema_version") != "tauvision-image-registry-v1":
        raise ValueError("unsupported tau-vision image registry")
    assets = registry.get("assets")
    if not isinstance(assets, dict):
        raise ValueError("release image registry has no assets mapping")

    tasks = [Task.model_validate(task) for task in raw_tasks]
    if task_split_name is not None:
        splits = get_tasks_split()
        if task_split_name not in splits:
            raise ValueError(
                f"Invalid task split name: {task_split_name}. Valid splits: {splits.keys()}"
            )
        selected = set(splits[task_split_name])
        tasks = [task for task in tasks if task.id in selected]
    if task_ids is not None:
        requested = set(task_ids)
        tasks = [task for task in tasks if task.id in requested]
        missing = requested - {task.id for task in tasks}
        if missing:
            raise ValueError(f"released tasks are missing requested IDs: {sorted(missing)}")
    if num_tasks is not None:
        tasks = tasks[:num_tasks]

    for task in tasks:
        task_assets: dict[str, dict[str, str]] = {}
        bank_by_index: dict[int, list[str]] = {}
        for entry in task.image_bank or []:
            kappa_index = entry.get("kappa_index")
            asset_ids = entry.get("asset_ids")
            if (
                type(kappa_index) is not int
                or not isinstance(asset_ids, list)
                or not asset_ids
                or kappa_index in bank_by_index
            ):
                raise ValueError(f"task {task.id} has an invalid image_bank entry")
            bank_by_index[kappa_index] = asset_ids
        for trigger in task.image_triggers or []:
            kappa_index = trigger.get("kappa_index")
            asset_ids = trigger.get("asset_ids")
            if asset_ids != bank_by_index.get(kappa_index):
                raise ValueError(
                    f"task {task.id} trigger assets disagree with image_bank"
                )
            for asset_id in asset_ids:
                if asset_id in task_assets:
                    continue
                try:
                    record = assets[asset_id]
                except KeyError as exc:
                    raise ValueError(
                        f"task {task.id} references unknown asset {asset_id}"
                    ) from exc
                raw_path = record.get("path")
                if not isinstance(raw_path, str) or not raw_path:
                    raise ValueError(f"asset {asset_id} has no path")
                path = (release_dir / raw_path).resolve()
                try:
                    path.relative_to(release_dir)
                except ValueError as exc:
                    raise ValueError(f"asset path escapes release: {raw_path}") from exc
                if not path.is_file():
                    raise ValueError(f"released asset is missing: {path}")
                data = path.read_bytes()
                digest = hashlib.sha256(data).hexdigest()
                if record.get("sha256") != digest:
                    raise ValueError(f"released asset hash mismatch: {asset_id}")
                if path.name != f"{digest}.png":
                    raise ValueError(f"released asset is not content-addressed: {asset_id}")
                task_assets[asset_id] = {
                    "image_content": base64.b64encode(data).decode("ascii"),
                    "alt_text": record.get(
                        "alt_text", "User-submitted product evidence photo."
                    ),
                }
        if task.image_triggers and not task_assets:
            raise ValueError(f"visual task {task.id} has no released image assets")
        task.runtime_image_assets = task_assets

    if not tasks:
        raise ValueError("published release yielded no tasks")
    return tasks
