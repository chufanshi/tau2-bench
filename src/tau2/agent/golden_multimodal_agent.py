"""Agent-only image injection for the frozen golden multimodal experiment.

The full task-specific English documents remain in ``domain_policy``. This
module validates a frozen sidecar bundle and adds the corresponding partial,
overlapping images to the agent's private initial LLM state. The envelope is
never added to the orchestrator trajectory or the user simulator state.
"""

import base64
import hashlib
import json
import os
import re
import struct
from pathlib import Path
from typing import Any, Optional

from tau2.agent.llm_agent import LLMAgent, LLMAgentStateType
from tau2.data_model.message import Message, UserMessage
from tau2.data_model.tasks import Task
from tau2.environment.tool import Tool

DATASET_ENV = "TAU2_GOLDEN_MM_DATASET"
ASSET_ROOT_ENV = "TAU2_GOLDEN_MM_ASSET_ROOT"

EXPECTED_SCHEMA_VERSION = "tau2-golden-multimodal-v1"
EXPECTED_SETTING = {
    "id": "golden_fulltext_partial_overlap_current_assets_v1",
    "text_mode": "full_original_text",
    "image_mode": "partial_overlapping_image",
    "semantic_union": "same_as_original_text",
    "exploratory": True,
}

GOLDEN_MULTIMODAL_ENVELOPE = (
    "These images are partial overlapping visual representations of the same "
    "task-specific golden documents already present in full English in the "
    "system policy. They add no authoritative facts. Use the full English text "
    "as the authority. Image order follows the required-document order."
)

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class GoldenMultimodalDatasetError(ValueError):
    """The frozen multimodal bundle does not match its declared provenance."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _canonical_json_sha256(value: Any) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return _sha256_text(serialized)


def _require_object(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise GoldenMultimodalDatasetError(f"{location} must be an object")
    return value


def _require_list(value: Any, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise GoldenMultimodalDatasetError(f"{location} must be a list")
    return value


def _require_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise GoldenMultimodalDatasetError(f"{location} must be a non-empty string")
    return value


def _require_sha256(value: Any, location: str) -> str:
    digest = _require_string(value, location)
    if _SHA256_PATTERN.fullmatch(digest) is None:
        raise GoldenMultimodalDatasetError(
            f"{location} must be a lowercase SHA-256 digest"
        )
    return digest


def _require_int(value: Any, location: str) -> int:
    if type(value) is not int or value <= 0:
        raise GoldenMultimodalDatasetError(f"{location} must be a positive integer")
    return value


def _read_png_dimensions(payload: bytes, location: str) -> tuple[int, int]:
    if len(payload) < 24 or payload[:8] != _PNG_SIGNATURE:
        raise GoldenMultimodalDatasetError(f"{location} is not a valid PNG file")
    if payload[12:16] != b"IHDR" or struct.unpack(">I", payload[8:12])[0] != 13:
        raise GoldenMultimodalDatasetError(
            f"{location} does not begin with a valid PNG IHDR chunk"
        )
    width, height = struct.unpack(">II", payload[16:24])
    if width <= 0 or height <= 0:
        raise GoldenMultimodalDatasetError(
            f"{location} declares invalid PNG dimensions {width}x{height}"
        )
    return width, height


def _default_asset_root(dataset_path: Path) -> Path:
    """Find the repository workspace without assuming the process cwd."""
    origins = (dataset_path.parent, Path.cwd(), Path(__file__).resolve().parent)
    for origin in origins:
        for candidate in (origin, *origin.parents):
            if (candidate / "tau2-bench").is_dir() and (
                candidate / "tauvision"
            ).is_dir():
                return candidate.resolve()
    raise GoldenMultimodalDatasetError(
        f"{ASSET_ROOT_ENV} is unset and the workspace root could not be inferred"
    )


def _load_dataset(dataset_path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GoldenMultimodalDatasetError(
            f"Could not read golden multimodal dataset {dataset_path}: {exc}"
        ) from exc
    return _require_object(raw, "dataset")


def _index_tasks(raw_tasks: Any) -> dict[str, dict[str, Any]]:
    """Index the canonical task list, accepting a keyed map for diagnostics."""
    indexed: dict[str, dict[str, Any]] = {}
    if isinstance(raw_tasks, list):
        items = [(None, item) for item in raw_tasks]
    elif isinstance(raw_tasks, dict):
        items = list(raw_tasks.items())
    else:
        raise GoldenMultimodalDatasetError("dataset.tasks must be a list or object")

    for index, (map_key, raw_task) in enumerate(items):
        bundle = _require_object(raw_task, f"dataset.tasks[{index}]")
        task_id = _require_string(
            bundle.get("task_id"), f"dataset.tasks[{index}].task_id"
        )
        if map_key is not None and map_key != task_id:
            raise GoldenMultimodalDatasetError(
                f"dataset.tasks key {map_key!r} does not match task_id {task_id!r}"
            )
        if task_id in indexed:
            raise GoldenMultimodalDatasetError(
                f"dataset contains duplicate task_id {task_id!r}"
            )
        indexed[task_id] = bundle
    return indexed


def _validate_setting(dataset: dict[str, Any]) -> dict[str, Any]:
    schema_version = _require_string(
        dataset.get("schema_version"), "dataset.schema_version"
    )
    if schema_version != EXPECTED_SCHEMA_VERSION:
        raise GoldenMultimodalDatasetError(
            f"Unsupported dataset.schema_version {schema_version!r}; "
            f"expected {EXPECTED_SCHEMA_VERSION!r}"
        )

    setting = _require_object(dataset.get("setting"), "dataset.setting")
    for field, expected in EXPECTED_SETTING.items():
        actual = setting.get(field)
        if type(actual) is not type(expected) or actual != expected:
            raise GoldenMultimodalDatasetError(
                f"dataset.setting.{field} is {actual!r}; expected {expected!r}"
            )
    _require_object(dataset.get("provenance"), "dataset.provenance")
    return setting


def _validate_bundle_hash(bundle: dict[str, Any], task_id: str) -> None:
    declared = _require_sha256(
        bundle.get("bundle_sha256"), f"task {task_id}.bundle_sha256"
    )
    hash_input = {key: value for key, value in bundle.items() if key != "bundle_sha256"}
    actual = _canonical_json_sha256(hash_input)
    if actual != declared:
        raise GoldenMultimodalDatasetError(
            f"task {task_id} bundle hash mismatch: expected {declared}, got {actual}"
        )


def _resolve_asset(asset_root: Path, asset_path: str, location: str) -> Path:
    relative_path = Path(asset_path)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise GoldenMultimodalDatasetError(
            f"{location} must be a contained relative path"
        )
    try:
        resolved = (asset_root / relative_path).resolve(strict=True)
    except OSError as exc:
        raise GoldenMultimodalDatasetError(
            f"{location} could not be resolved: {exc}"
        ) from exc
    try:
        resolved.relative_to(asset_root)
    except ValueError as exc:
        raise GoldenMultimodalDatasetError(
            f"{location} escapes asset root {asset_root}"
        ) from exc
    if not resolved.is_file():
        raise GoldenMultimodalDatasetError(f"{location} is not a regular file")
    return resolved


def _load_task_images(
    *,
    dataset: dict[str, Any],
    asset_root: Path,
    task: Task,
    domain_policy: str,
) -> list[str]:
    setting = _validate_setting(dataset)
    indexed_tasks = _index_tasks(dataset.get("tasks"))
    if task.id not in indexed_tasks:
        raise GoldenMultimodalDatasetError(
            f"task_id {task.id!r} is absent from the golden multimodal dataset"
        )
    bundle = indexed_tasks[task.id]
    _validate_bundle_hash(bundle, task.id)

    policy_sha256 = _require_sha256(
        bundle.get("policy_sha256"), f"task {task.id}.policy_sha256"
    )
    actual_policy_sha256 = _sha256_text(domain_policy)
    if policy_sha256 != actual_policy_sha256:
        raise GoldenMultimodalDatasetError(
            f"task {task.id} policy hash mismatch: expected {policy_sha256}, "
            f"got {actual_policy_sha256}"
        )
    _require_sha256(
        bundle.get("required_documents_block_sha256"),
        f"task {task.id}.required_documents_block_sha256",
    )

    required_documents_raw = _require_list(
        bundle.get("required_documents"), f"task {task.id}.required_documents"
    )
    required_documents = [
        _require_string(item, f"task {task.id}.required_documents[{index}]")
        for index, item in enumerate(required_documents_raw)
    ]
    task_required_documents = list(task.required_documents or [])
    if required_documents != task_required_documents:
        raise GoldenMultimodalDatasetError(
            f"task {task.id} required-document order does not match the frozen bundle"
        )

    raw_documents = _require_list(bundle.get("documents"), f"task {task.id}.documents")
    if len(raw_documents) != len(required_documents):
        raise GoldenMultimodalDatasetError(
            f"task {task.id} has {len(raw_documents)} document records for "
            f"{len(required_documents)} required documents"
        )

    image_pages: list[str] = []
    for index, (raw_document, expected_document_id) in enumerate(
        zip(raw_documents, required_documents, strict=True)
    ):
        location = f"task {task.id}.documents[{index}]"
        document = _require_object(raw_document, location)

        ordinal = document.get("ordinal")
        if type(ordinal) is not int or ordinal != index:
            raise GoldenMultimodalDatasetError(
                f"{location}.ordinal is {ordinal!r}; expected {index}"
            )
        document_id = _require_string(
            document.get("document_id"), f"{location}.document_id"
        )
        if document_id != expected_document_id:
            raise GoldenMultimodalDatasetError(
                f"{location}.document_id is {document_id!r}; "
                f"expected {expected_document_id!r}"
            )

        title = _require_string(document.get("title"), f"{location}.title")
        title_sha256 = _require_sha256(
            document.get("title_sha256"), f"{location}.title_sha256"
        )
        if _sha256_text(title) != title_sha256:
            raise GoldenMultimodalDatasetError(f"{location} title hash mismatch")
        _require_sha256(document.get("content_sha256"), f"{location}.content_sha256")

        for mode_field in ("text_mode", "image_mode"):
            expected_mode = setting[mode_field]
            actual_mode = document.get(mode_field)
            if actual_mode != expected_mode:
                raise GoldenMultimodalDatasetError(
                    f"{location}.{mode_field} is {actual_mode!r}; "
                    f"expected {expected_mode!r}"
                )

        source_anchors_raw = _require_list(
            document.get("source_anchors"), f"{location}.source_anchors"
        )
        source_anchors = [
            _require_string(anchor, f"{location}.source_anchors[{anchor_index}]")
            for anchor_index, anchor in enumerate(source_anchors_raw)
        ]
        if not source_anchors:
            raise GoldenMultimodalDatasetError(
                f"{location}.source_anchors must not be empty"
            )
        source_anchors_sha256 = _require_sha256(
            document.get("source_anchors_sha256"),
            f"{location}.source_anchors_sha256",
        )
        if _canonical_json_sha256(source_anchors) != source_anchors_sha256:
            raise GoldenMultimodalDatasetError(
                f"{location} source-anchor hash mismatch"
            )

        semantic_verdict = _require_string(
            document.get("semantic_verdict"), f"{location}.semantic_verdict"
        )
        if semantic_verdict not in {"PASS", "WARN"}:
            raise GoldenMultimodalDatasetError(
                f"{location}.semantic_verdict {semantic_verdict!r} is not usable"
            )
        if "audit_provenance" not in document:
            raise GoldenMultimodalDatasetError(
                f"{location}.audit_provenance is required"
            )

        mime_type = _require_string(document.get("mime_type"), f"{location}.mime_type")
        if mime_type != "image/png":
            raise GoldenMultimodalDatasetError(
                f"{location}.mime_type is {mime_type!r}; expected 'image/png'"
            )
        asset_path = _require_string(
            document.get("asset_path"), f"{location}.asset_path"
        )
        asset_sha256 = _require_sha256(
            document.get("asset_sha256"), f"{location}.asset_sha256"
        )
        declared_width = _require_int(document.get("width"), f"{location}.width")
        declared_height = _require_int(document.get("height"), f"{location}.height")

        resolved_asset = _resolve_asset(
            asset_root, asset_path, f"{location}.asset_path"
        )
        try:
            payload = resolved_asset.read_bytes()
        except OSError as exc:
            raise GoldenMultimodalDatasetError(
                f"Could not read {location}.asset_path {resolved_asset}: {exc}"
            ) from exc
        actual_asset_sha256 = _sha256_bytes(payload)
        if actual_asset_sha256 != asset_sha256:
            raise GoldenMultimodalDatasetError(
                f"{location} asset hash mismatch: expected {asset_sha256}, "
                f"got {actual_asset_sha256}"
            )
        actual_width, actual_height = _read_png_dimensions(payload, str(resolved_asset))
        if (actual_width, actual_height) != (declared_width, declared_height):
            raise GoldenMultimodalDatasetError(
                f"{location} image dimensions are {actual_width}x{actual_height}; "
                f"expected {declared_width}x{declared_height}"
            )
        image_pages.append(base64.b64encode(payload).decode("ascii"))

    if not image_pages:
        raise GoldenMultimodalDatasetError(
            f"task {task.id} does not contain any golden multimodal images"
        )
    return image_pages


class GoldenMultimodalAgent(LLMAgent):
    """LLM agent with validated, task-specific image context in private state."""

    def __init__(
        self,
        tools: list[Tool],
        domain_policy: str,
        task: Task,
        llm: str,
        llm_args: Optional[dict] = None,
        dataset_path: Optional[Path] = None,
        asset_root: Optional[Path] = None,
    ):
        super().__init__(
            tools=tools,
            domain_policy=domain_policy,
            llm=llm,
            llm_args=llm_args,
        )
        if task is None:
            raise GoldenMultimodalDatasetError(
                "GoldenMultimodalAgent requires the current task"
            )

        if dataset_path is None:
            raw_dataset_path = os.environ.get(DATASET_ENV)
            if not raw_dataset_path:
                raise GoldenMultimodalDatasetError(
                    f"{DATASET_ENV} must point to the frozen multimodal dataset"
                )
            dataset_path = Path(raw_dataset_path)
        try:
            resolved_dataset_path = dataset_path.expanduser().resolve(strict=True)
        except OSError as exc:
            raise GoldenMultimodalDatasetError(
                f"Could not resolve {DATASET_ENV} path {dataset_path}: {exc}"
            ) from exc
        if not resolved_dataset_path.is_file():
            raise GoldenMultimodalDatasetError(
                f"{DATASET_ENV} path is not a file: {resolved_dataset_path}"
            )

        if asset_root is None:
            raw_asset_root = os.environ.get(ASSET_ROOT_ENV)
            asset_root = (
                Path(raw_asset_root)
                if raw_asset_root
                else _default_asset_root(resolved_dataset_path)
            )
        try:
            resolved_asset_root = asset_root.expanduser().resolve(strict=True)
        except OSError as exc:
            raise GoldenMultimodalDatasetError(
                f"Could not resolve {ASSET_ROOT_ENV} path {asset_root}: {exc}"
            ) from exc
        if not resolved_asset_root.is_dir():
            raise GoldenMultimodalDatasetError(
                f"{ASSET_ROOT_ENV} path is not a directory: {resolved_asset_root}"
            )

        dataset = _load_dataset(resolved_dataset_path)
        self._image_pages = _load_task_images(
            dataset=dataset,
            asset_root=resolved_asset_root,
            task=task,
            domain_policy=domain_policy,
        )
        self.task_id = task.id
        self.dataset_path = resolved_dataset_path
        self.asset_root = resolved_asset_root

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> LLMAgentStateType:
        # Never insert the private envelope into a caller-owned history list.
        private_history = list(message_history) if message_history is not None else None
        state = super().get_init_state(message_history=private_history)
        envelope = UserMessage(
            role="user",
            content=GOLDEN_MULTIMODAL_ENVELOPE,
            image_pages=list(self._image_pages),
        )
        # This state is private to the agent. Orchestrator trajectories are built
        # independently from actual participant/environment messages, so neither
        # the envelope nor its images enter results or the user simulator state.
        state.messages.insert(0, envelope)
        return state


def create_golden_multimodal_agent(tools, domain_policy, **kwargs):
    """Factory for the frozen full-text + partial-overlap-image agent."""
    task = kwargs.get("task")
    if task is None:
        raise GoldenMultimodalDatasetError(
            "llm_agent_golden_multimodal requires a task"
        )
    return GoldenMultimodalAgent(
        tools=tools,
        domain_policy=domain_policy,
        task=task,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
    )
