"""Fail-closed runtime loader for the boundary-refined v3 overlay."""

from __future__ import annotations

import hashlib
import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tau2.data_model.tasks import Task


MANIFEST_ENV = "TAU2_COMPLEMENTARY_INPLACE_MANIFEST"
ASSET_ROOT_ENV = "TAU2_COMPLEMENTARY_INPLACE_ASSET_ROOT"
ARM_ENV = "TAU2_COMPLEMENTARY_INPLACE_ARM"
SCHEMA_VERSION = "tauvision-complementary-inplace-asset-manifest-v3"
SETTING_ID = "complementary_inplace_system_v3"
# SHA-256 of the reviewed, frozen v3 asset manifest.
EXPECTED_MANIFEST_SHA256 = "0905556d5ed2af8e0089db37b1a7ea6344ca76c12aefd2aa55507cbd0e452660"
EXPECTED_SELECTION_SHA256 = "deea5b4375a58ba674e439254002d22a1b5f653a7fd49bd46e915e796a6f686f"
EXPECTED_FORMS = {"B": 13, "B_S": 24, "P_B": 13, "P_B_S": 10}
EXPECTED_ORIGINS = {"generated_v3": 16, "reused_exact_v2": 44}
EXPECTED_TASK_COUNT = 19
EXPECTED_DOCUMENT_COUNT = 60
EXPECTED_DOCUMENT_REFERENCE_COUNT = 109
ARMS = ("inplace_text", "inplace_image")
FROZEN_TASK_IDS = (
    "task_001", "task_004", "task_006", "task_007", "task_016",
    "task_017", "task_021", "task_022", "task_024", "task_025",
    "task_032", "task_035", "task_036", "task_050", "task_052",
    "task_055", "task_058", "task_059", "task_064",
)
_DEFAULT_MANIFEST = Path(
    "tauvision/data/banking_explorer/complementary_inplace_system_v3/asset_manifest.json"
)


class ComplementaryInplaceSystemV3Error(ValueError):
    """A frozen v3 runtime invariant failed."""


@dataclass(frozen=True)
class RuntimeDocument:
    document_id: str
    document_key: str
    title: str
    source_content: str
    prefix: str
    block: str
    suffix: str
    char_start: int
    char_end: int
    decomposition_form: str
    image_payload: bytes
    image_sha256: str
    image_width: int
    image_height: int
    image_path: Path


@dataclass(frozen=True)
class RuntimeTask:
    arm: str
    task_id: str
    domain_policy: str
    documents: tuple[RuntimeDocument, ...]
    manifest_path: Path
    asset_root: Path


def _fail(message: str) -> None:
    raise ComplementaryInplaceSystemV3Error(message)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _text_sha(value: str) -> str:
    return _sha(value.encode("utf-8"))


def _object(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{location} must be an object")
    return value


def _rows(value: Any, location: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(f"{location} must be a list")
    return value


def _load(path: Path, location: str) -> tuple[bytes, dict[str, Any]]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(f"cannot load {location} at {path}: {exc}")
    return payload, _object(value, location)


def _infer_root(origin: Path | None = None) -> Path:
    for start in (origin, Path.cwd(), Path(__file__).resolve().parent):
        if start is None:
            continue
        for candidate in (start, *start.parents):
            if (candidate / "tau2-bench").is_dir() and (candidate / "tauvision").is_dir():
                return candidate.resolve()
    _fail(f"{ASSET_ROOT_ENV} is unset and the workspace root cannot be inferred")


def resolve_asset_root(path: Path | None = None) -> Path:
    raw = path or (Path(os.environ[ASSET_ROOT_ENV]) if os.environ.get(ASSET_ROOT_ENV) else None)
    root = (raw or _infer_root()).expanduser().resolve(strict=True)
    if not root.is_dir() or not (root / "tau2-bench").is_dir() or not (root / "tauvision").is_dir():
        _fail(f"invalid Tau-Vision asset root: {root}")
    return root


def _contained(root: Path, raw: Any, location: str) -> Path:
    if not isinstance(raw, str) or not raw:
        _fail(f"{location} must be a nonempty workspace-relative path")
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        _fail(f"{location} is not contained")
    path = (root / relative).resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError:
        _fail(f"{location} escapes the asset root")
    if not path.is_file():
        _fail(f"{location} is not a regular file")
    return path


def _bound_json(root: Path, ref: dict[str, Any], location: str) -> tuple[Path, dict[str, Any]]:
    path = _contained(root, ref.get("path"), location)
    payload, value = _load(path, location)
    if _sha(payload) != ref.get("sha256"):
        _fail(f"{location} hash mismatch")
    return path, value


def _index(rows: list[Any], location: str) -> dict[str, dict[str, Any]]:
    result = {}
    for raw in rows:
        row = _object(raw, location)
        identifier = row.get("document_id")
        if not isinstance(identifier, str) or identifier in result:
            _fail(f"{location} has an invalid/duplicate document_id")
        result[identifier] = row
    return result


def resolve_manifest_path(path: Path | None = None, *, asset_root: Path | None = None) -> tuple[Path, Path]:
    root = resolve_asset_root(asset_root)
    configured = os.environ.get(MANIFEST_ENV)
    raw = path or (Path(configured) if configured else _DEFAULT_MANIFEST)
    candidate = raw if raw.is_absolute() else root / raw
    resolved = candidate.expanduser().resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError:
        _fail("runtime manifest escapes asset root")
    if not resolved.is_file():
        _fail("runtime manifest is not a file")
    return resolved, root


def resolve_arm(arm: str | None = None) -> str:
    value = arm or os.environ.get(ARM_ENV)
    if value not in ARMS:
        _fail(f"{ARM_ENV} must be one of {list(ARMS)}, got {value!r}")
    return value


def _validate_global(manifest: dict[str, Any], root: Path) -> tuple[dict, dict, dict, dict]:
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("setting_id") != SETTING_ID:
        _fail("manifest schema/setting mismatch")
    if manifest.get("status") != "frozen_run_eligible":
        _fail("manifest is not run eligible")
    counts = _object(manifest.get("counts"), "manifest.counts")
    if counts.get("documents") != 60 or counts.get("tasks") != 19 or counts.get("document_references") != 109:
        _fail("manifest closure counts mismatch")
    if counts.get("forms") != EXPECTED_FORMS or counts.get("origins") != EXPECTED_ORIGINS:
        _fail("manifest form/origin counts mismatch")
    entries = _index(_rows(manifest.get("entries"), "manifest.entries"), "manifest.entries")
    bindings = _object(manifest.get("bindings"), "manifest.bindings")
    _, worklist = _bound_json(root, {"path": bindings.get("worklist_path"), "sha256": bindings.get("worklist_sha256")}, "worklist")
    _, bundle = _bound_json(root, {"path": bindings.get("annotation_bundle_path"), "sha256": bindings.get("annotation_bundle_sha256")}, "annotation bundle")
    if worklist.get("schema_version") != "tauvision-complementary-inplace-worklist-v3" or worklist.get("setting_id") != SETTING_ID:
        _fail("worklist schema/setting mismatch")
    if bundle.get("schema_version") != "tauvision-complementary-inplace-annotation-bundle-v3" or bundle.get("setting_id") != SETTING_ID or bundle.get("status") != "frozen_run_eligible":
        _fail("bundle schema/setting/status mismatch")
    work_by_id = _index(_rows(worklist.get("items"), "worklist.items"), "worklist.items")
    bundle_by_id = _index(_rows(bundle.get("documents"), "bundle.documents"), "bundle.documents")
    if set(entries) != set(work_by_id) or set(entries) != set(bundle_by_id) or len(entries) != 60:
        _fail("manifest/worklist/bundle document closure mismatch")
    selection_ref = _object(worklist.get("selection"), "worklist.selection")
    if selection_ref.get("sha256") != EXPECTED_SELECTION_SHA256:
        _fail("selection hash differs from frozen cohort")
    _, selection = _bound_json(root, selection_ref, "selection")
    if selection.get("schema_version") != "tauvision-complementary-selection-v1":
        _fail("selection schema mismatch")
    task_rows = _rows(selection.get("tasks"), "selection.tasks")
    task_by_id = {}
    for row in task_rows:
        row = _object(row, "selection task")
        identifier = row.get("task_id")
        if identifier in task_by_id:
            _fail("duplicate selection task")
        task_by_id[identifier] = row
    if tuple(selection.get("task_ids", [])) != FROZEN_TASK_IDS or tuple(task_by_id) != FROZEN_TASK_IDS:
        _fail("selection task order differs from frozen cohort")
    return entries, work_by_id, bundle_by_id, task_by_id


def _load_document(*, task_id: str, entry: dict, work: dict, bundle: dict, root: Path) -> RuntimeDocument:
    document_id = entry["document_id"]
    if task_id not in entry.get("task_ids", []):
        _fail(f"{document_id} lacks task binding")
    if {row.get("task_id") for row in work.get("task_contexts", [])} != set(entry.get("task_ids", [])):
        _fail(f"{document_id} worklist task bindings differ")
    annotation_path, annotation = _bound_json(root, _object(entry.get("annotation"), "entry.annotation"), "annotation")
    if bundle.get("annotation_path") != entry["annotation"]["path"] or bundle.get("annotation_sha256") != entry["annotation"]["sha256"]:
        _fail(f"{document_id} bundle annotation binding mismatch")
    if annotation.get("schema_version") != "tauvision-complementary-inplace-document-v3" or annotation.get("setting_id") != SETTING_ID or annotation.get("status") != "independent_review_passed":
        _fail(f"{document_id} annotation schema/setting/status mismatch")
    source_ref = _object(annotation.get("source"), "annotation.source")
    source_path = _contained(root, source_ref.get("path"), "source")
    source_payload, source = _load(source_path, "source")
    if _sha(source_payload) != source_ref.get("file_sha256"):
        _fail(f"{document_id} source file hash mismatch")
    content = source.get("content")
    title = source.get("title")
    if not isinstance(content, str) or not isinstance(title, str) or _text_sha(content) != entry.get("source_content_sha256"):
        _fail(f"{document_id} source content mismatch")
    block = _object(annotation.get("block"), "annotation.block")
    start, end = block.get("char_start"), block.get("char_end")
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(content):
        _fail(f"{document_id} invalid block offsets")
    prefix, selected, suffix = content[:start], content[start:end], content[end:]
    form = "B" if not prefix and not suffix else "B_S" if not prefix else "P_B" if not suffix else "P_B_S"
    if selected != block.get("exact_text") or _text_sha(selected) != entry.get("block_sha256") or block.get("decomposition_form") != form or entry.get("decomposition_form") != form:
        _fail(f"{document_id} block/form binding mismatch")
    if _text_sha(prefix + selected + suffix) != source_ref.get("content_sha256"):
        _fail(f"{document_id} P+B+S reconstruction mismatch")
    asset = _object(entry.get("asset"), "entry.asset")
    image_path = _contained(root, asset.get("path"), "asset")
    image = image_path.read_bytes()
    if _sha(image) != asset.get("sha256") or image[:8] != b"\x89PNG\r\n\x1a\n" or image[12:16] != b"IHDR":
        _fail(f"{document_id} asset hash/format mismatch")
    width, height = struct.unpack(">II", image[16:24])
    if (width, height) != (asset.get("width"), asset.get("height")):
        _fail(f"{document_id} asset dimensions mismatch")
    if annotation_path.name != Path(entry["annotation"]["path"]).name:
        _fail(f"{document_id} annotation filename mismatch")
    return RuntimeDocument(
        document_id=document_id, document_key=entry["document_key"], title=title,
        source_content=content, prefix=prefix, block=selected, suffix=suffix,
        char_start=start, char_end=end, decomposition_form=form,
        image_payload=image, image_sha256=asset["sha256"], image_width=width,
        image_height=height, image_path=image_path,
    )


def load_task_runtime(*, task: Task, domain_policy: str, arm: str | None = None, manifest_path: Path | None = None, asset_root: Path | None = None) -> RuntimeTask:
    selected_arm = resolve_arm(arm)
    path, root = resolve_manifest_path(manifest_path, asset_root=asset_root)
    manifest_payload, manifest = _load(path, "runtime manifest")
    if _sha(manifest_payload) != EXPECTED_MANIFEST_SHA256:
        _fail("runtime manifest hash differs from the frozen v3 release")
    entries, work_by_id, bundle_by_id, selection_by_task = _validate_global(manifest, root)
    if task.id not in selection_by_task:
        _fail(f"task {task.id!r} is outside the frozen cohort")
    task_row = selection_by_task[task.id]
    required = task.required_documents or []
    if required != task_row.get("required_documents") or not required:
        _fail(f"task {task.id} required-document order differs from the frozen cohort")
    manifest_docs = {key for key, value in entries.items() if task.id in value.get("task_ids", [])}
    if manifest_docs != set(required):
        _fail(f"task {task.id} manifest document set mismatch")
    documents = tuple(
        _load_document(task_id=task.id, entry=entries[key], work=work_by_id[key], bundle=bundle_by_id[key], root=root)
        for key in required
    )
    from tau2.domains.banking_knowledge.retrieval import PROMPTS_DIR, load_prompt_template
    required_text = "\n\n---\n\n".join(f"## {doc.title}\n\n{doc.source_content}" for doc in documents)
    expected_policy = load_prompt_template(PROMPTS_DIR / "required_docs.md", knowledge_base=None).replace("{{required_documents}}", required_text)
    if domain_policy != expected_policy:
        _fail(f"task {task.id} policy is not byte-identical Golden Retrieval text")
    return RuntimeTask(arm=selected_arm, task_id=task.id, domain_policy=domain_policy, documents=documents, manifest_path=path, asset_root=root)
