"""Frozen runtime for EXP4 in-place system-policy image replacement.

The adapter intentionally leaves the canonical Golden Retrieval policy as
plain text until agent construction.  The text arm sends that string without
modification.  The image arm replaces each reviewed continuous block at its
exact character offsets while preserving every other character and the task's
required-document order.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tau2.data_model.tasks import Task

MANIFEST_ENV = "TAU2_COMPLEMENTARY_INPLACE_MANIFEST"
ASSET_ROOT_ENV = "TAU2_COMPLEMENTARY_INPLACE_ASSET_ROOT"
ARM_ENV = "TAU2_COMPLEMENTARY_INPLACE_ARM"

SCHEMA_VERSION = "tauvision-complementary-inplace-asset-manifest-v2"
SETTING_ID = "complementary_inplace_system_v2"
EXPECTED_MANIFEST_SHA256 = (
    "8566a1eb903252e8a5c3b1fab1b11eab2d87a42f6ad6a7eab9afcdca0c911f5f"
)
EXPECTED_SELECTION_SHA256 = (
    "deea5b4375a58ba674e439254002d22a1b5f653a7fd49bd46e915e796a6f686f"
)
EXPECTED_TASK_COUNT = 19
EXPECTED_DOCUMENT_COUNT = 60
EXPECTED_DOCUMENT_REFERENCE_COUNT = 109
ARMS = ("inplace_text", "inplace_image")
FORMS = ("B", "B_S", "P_B", "P_B_S")
FROZEN_TASK_IDS = (
    "task_001",
    "task_004",
    "task_006",
    "task_007",
    "task_016",
    "task_017",
    "task_021",
    "task_022",
    "task_024",
    "task_025",
    "task_032",
    "task_035",
    "task_036",
    "task_050",
    "task_052",
    "task_055",
    "task_058",
    "task_059",
    "task_064",
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_DEFAULT_MANIFEST = Path(
    "tauvision/data/banking_explorer/complementary_inplace_system_v2/"
    "asset_manifest.json"
)


class ComplementaryInplaceSystemError(ValueError):
    """A frozen in-place system-policy invariant was not satisfied."""


@dataclass(frozen=True)
class RuntimeDocument:
    """One fully validated source block and its reviewed image realization."""

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
    """Validated material for one selected task and experimental arm."""

    arm: str
    task_id: str
    domain_policy: str
    documents: tuple[RuntimeDocument, ...]
    manifest_path: Path
    asset_root: Path


def _fail(message: str) -> None:
    raise ComplementaryInplaceSystemError(message)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _object(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{location} must be an object")
    return value


def _list(value: Any, location: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(f"{location} must be a list")
    return value


def _string(value: Any, location: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        _fail(
            f"{location} must be a string{' (possibly empty)' if allow_empty else ''}"
        )
    return value


def _sha256(value: Any, location: str) -> str:
    digest = _string(value, location)
    if _SHA256_RE.fullmatch(digest) is None:
        _fail(f"{location} must be a lowercase SHA-256 digest")
    return digest


def _positive_int(value: Any, location: str) -> int:
    if type(value) is not int or value <= 0:
        _fail(f"{location} must be a positive integer")
    return value


def _nonnegative_int(value: Any, location: str) -> int:
    if type(value) is not int or value < 0:
        _fail(f"{location} must be a non-negative integer")
    return value


def _load_json_bytes(path: Path, location: str) -> tuple[bytes, dict[str, Any]]:
    try:
        payload = path.read_bytes()
        parsed = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(f"cannot read {location} at {path}: {exc}")
    return payload, _object(parsed, location)


def _infer_asset_root(origin: Path | None = None) -> Path:
    origins = [origin, Path.cwd(), Path(__file__).resolve().parent]
    for start in (item for item in origins if item is not None):
        for candidate in (start, *start.parents):
            if (candidate / "tau2-bench").is_dir() and (
                candidate / "tauvision"
            ).is_dir():
                return candidate.resolve()
    _fail(f"{ASSET_ROOT_ENV} is unset and the workspace root cannot be inferred")


def resolve_asset_root(path: Path | None = None) -> Path:
    """Resolve the workspace root used by every contained manifest path."""
    if path is None:
        raw = os.environ.get(ASSET_ROOT_ENV)
        path = Path(raw) if raw else _infer_asset_root()
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as exc:
        _fail(f"cannot resolve asset root {path}: {exc}")
    if not resolved.is_dir():
        _fail(f"asset root is not a directory: {resolved}")
    if not (resolved / "tau2-bench").is_dir() or not (resolved / "tauvision").is_dir():
        _fail(f"asset root does not look like the Tau-Vision workspace: {resolved}")
    return resolved


def _resolve_contained(root: Path, raw_path: str, location: str) -> Path:
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        _fail(f"{location} must be a contained workspace-relative path")
    try:
        resolved = (root / relative).resolve(strict=True)
    except OSError as exc:
        _fail(f"cannot resolve {location}: {exc}")
    try:
        resolved.relative_to(root)
    except ValueError:
        _fail(f"{location} escapes asset root {root}")
    if not resolved.is_file():
        _fail(f"{location} is not a regular file")
    return resolved


def resolve_manifest_path(
    path: Path | None = None, *, asset_root: Path | None = None
) -> tuple[Path, Path]:
    """Resolve the immutable v2 manifest and its workspace root."""
    raw = path
    if raw is None:
        configured = os.environ.get(MANIFEST_ENV)
        raw = Path(configured) if configured else _DEFAULT_MANIFEST
    root = resolve_asset_root(asset_root)
    candidate = raw if raw.is_absolute() else root / raw
    try:
        resolved = candidate.expanduser().resolve(strict=True)
    except OSError as exc:
        _fail(f"cannot resolve runtime manifest {candidate}: {exc}")
    try:
        resolved.relative_to(root)
    except ValueError:
        _fail(f"runtime manifest must be contained by asset root {root}")
    if not resolved.is_file():
        _fail(f"runtime manifest is not a regular file: {resolved}")
    return resolved, root


def resolve_arm(arm: str | None = None) -> str:
    """Resolve an explicit matched experimental arm."""
    value = arm or os.environ.get(ARM_ENV)
    if value not in ARMS:
        _fail(f"{ARM_ENV} must be one of {list(ARMS)}, got {value!r}")
    return value


def _bound_json(
    root: Path, raw_path: Any, expected_sha: Any, location: str
) -> tuple[Path, dict[str, Any]]:
    path = _resolve_contained(root, _string(raw_path, f"{location}.path"), location)
    payload, parsed = _load_json_bytes(path, location)
    if _sha256_bytes(payload) != _sha256(expected_sha, f"{location}.sha256"):
        _fail(f"{location} file hash mismatch")
    return path, parsed


def _png_dimensions(payload: bytes, location: str) -> tuple[int, int]:
    if len(payload) < 24 or payload[:8] != _PNG_SIGNATURE:
        _fail(f"{location} is not a PNG")
    if payload[12:16] != b"IHDR" or struct.unpack(">I", payload[8:12])[0] != 13:
        _fail(f"{location} has no canonical first IHDR chunk")
    width, height = struct.unpack(">II", payload[16:24])
    if width <= 0 or height <= 0:
        _fail(f"{location} has invalid dimensions {width}x{height}")
    return width, height


def _index_unique(
    rows: list[Any], key: str, location: str
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for index, raw_row in enumerate(rows):
        row_location = f"{location}[{index}]"
        row = _object(raw_row, row_location)
        identifier = _string(row.get(key), f"{row_location}.{key}")
        if identifier in result:
            _fail(f"{location} contains duplicate {key} {identifier!r}")
        result[identifier] = row
    return result


def _validate_global_bundle(
    manifest: dict[str, Any], root: Path
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        _fail("runtime manifest schema version mismatch")
    if manifest.get("setting_id") != SETTING_ID:
        _fail("runtime manifest setting ID mismatch")
    if manifest.get("status") != "frozen_run_eligible":
        _fail("runtime manifest is not frozen_run_eligible")

    counts = _object(manifest.get("counts"), "manifest.counts")
    if counts.get("documents") != EXPECTED_DOCUMENT_COUNT:
        _fail("runtime manifest document count mismatch")
    if counts.get("tasks") != EXPECTED_TASK_COUNT:
        _fail("runtime manifest task count mismatch")
    if counts.get("forms") != {"B": 29, "B_S": 16, "P_B": 5, "P_B_S": 10}:
        _fail("runtime manifest decomposition-form counts mismatch")

    entries = _list(manifest.get("entries"), "manifest.entries")
    if len(entries) != EXPECTED_DOCUMENT_COUNT:
        _fail("runtime manifest must contain exactly 60 entries")
    manifest_by_id = _index_unique(entries, "document_id", "manifest.entries")

    bindings = _object(manifest.get("bindings"), "manifest.bindings")
    _, worklist = _bound_json(
        root,
        bindings.get("worklist_path"),
        bindings.get("worklist_sha256"),
        "manifest.bindings.worklist",
    )
    _, bundle = _bound_json(
        root,
        bindings.get("annotation_bundle_path"),
        bindings.get("annotation_bundle_sha256"),
        "manifest.bindings.annotation_bundle",
    )
    if bundle.get("schema_version") != (
        "tauvision-complementary-inplace-annotation-bundle-v2"
    ):
        _fail("annotation bundle schema version mismatch")
    if bundle.get("setting_id") != SETTING_ID:
        _fail("annotation bundle setting ID mismatch")
    if bundle.get("status") != "frozen_run_eligible":
        _fail("annotation bundle is not frozen_run_eligible")
    bundle_rows = _list(bundle.get("documents"), "annotation_bundle.documents")
    if len(bundle_rows) != EXPECTED_DOCUMENT_COUNT:
        _fail("annotation bundle must contain exactly 60 documents")
    bundle_by_id = _index_unique(
        bundle_rows, "document_id", "annotation_bundle.documents"
    )
    if set(bundle_by_id) != set(manifest_by_id):
        _fail("manifest and annotation bundle document sets differ")

    if worklist.get("schema_version") != (
        "tauvision-complementary-inplace-worklist-v2"
    ):
        _fail("worklist schema version mismatch")
    if worklist.get("setting_id") != SETTING_ID:
        _fail("worklist setting ID mismatch")
    worklist_rows = _list(worklist.get("items"), "worklist.items")
    if len(worklist_rows) != EXPECTED_DOCUMENT_COUNT:
        _fail("worklist must contain exactly 60 documents")
    worklist_by_id = _index_unique(worklist_rows, "document_id", "worklist.items")
    if set(worklist_by_id) != set(manifest_by_id):
        _fail("manifest and worklist document sets differ")

    selection_binding = _object(worklist.get("selection"), "worklist.selection")
    if selection_binding.get("sha256") != EXPECTED_SELECTION_SHA256:
        _fail("worklist selection binding is not the frozen 5-of-5 cohort")
    _, selection = _bound_json(
        root,
        selection_binding.get("path"),
        selection_binding.get("sha256"),
        "worklist.selection",
    )
    if selection.get("schema_version") != "tauvision-complementary-selection-v1":
        _fail("selection schema version mismatch")
    status = _object(selection.get("status"), "selection.status")
    required_status = (
        "state",
        "validated",
        "task_sets_aligned",
        "all_runs_complete",
        "all_runs_log_validated_rc0",
        "all_rewards_binary_and_non_null",
        "all_selected_rewards_equal_one",
        "deterministic_output",
    )
    if status.get("state") != "frozen" or not all(
        status.get(field) is True for field in required_status[1:]
    ):
        _fail("selection is not a fully validated frozen cohort")
    selection_counts = _object(selection.get("counts"), "selection.counts")
    if (
        selection_counts.get("selected_tasks") != EXPECTED_TASK_COUNT
        or selection_counts.get("unique_required_documents") != EXPECTED_DOCUMENT_COUNT
        or selection_counts.get("required_document_references")
        != EXPECTED_DOCUMENT_REFERENCE_COUNT
    ):
        _fail("selection counts mismatch")
    task_rows = _list(selection.get("tasks"), "selection.tasks")
    if len(task_rows) != EXPECTED_TASK_COUNT:
        _fail("selection must contain exactly 19 tasks")
    selection_by_task = _index_unique(task_rows, "task_id", "selection.tasks")
    selected_ids = _list(selection.get("task_ids"), "selection.task_ids")
    if selected_ids != list(FROZEN_TASK_IDS):
        _fail("selection task IDs differ from the frozen v2 cohort")
    if selected_ids != list(selection_by_task):
        _fail("selection task IDs and task rows differ or are out of order")
    if selection_binding.get("task_ids") != selected_ids:
        _fail("worklist and selection task IDs differ")
    return manifest_by_id, bundle_by_id, worklist_by_id, selection_by_task


def _validate_source_and_block(
    *,
    task_id: str,
    entry: dict[str, Any],
    bundle_row: dict[str, Any],
    worklist_row: dict[str, Any],
    root: Path,
) -> RuntimeDocument:
    document_id = _string(entry.get("document_id"), "manifest entry.document_id")
    location = f"document {document_id}"
    document_key = _string(entry.get("document_key"), f"{location}.document_key")
    if bundle_row.get("document_key") != document_key:
        _fail(f"{location} bundle document key mismatch")
    if worklist_row.get("document_key") != document_key:
        _fail(f"{location} worklist document key mismatch")

    entry_task_ids = _list(entry.get("task_ids"), f"{location}.task_ids")
    if task_id not in entry_task_ids or len(entry_task_ids) != len(set(entry_task_ids)):
        _fail(f"{location} task binding is absent or duplicated")
    contexts = _list(worklist_row.get("task_contexts"), f"{location}.task_contexts")
    context_ids = [
        _string(_object(row, f"{location}.task_contexts").get("task_id"), "task_id")
        for row in contexts
    ]
    if task_id not in context_ids or set(context_ids) != set(entry_task_ids):
        _fail(f"{location} manifest/worklist task bindings differ")

    annotation_ref = _object(entry.get("annotation"), f"{location}.annotation")
    annotation_path, annotation = _bound_json(
        root,
        annotation_ref.get("path"),
        annotation_ref.get("sha256"),
        f"{location}.annotation",
    )
    if bundle_row.get("annotation_path") != annotation_ref.get("path"):
        _fail(f"{location} annotation bundle path mismatch")
    if bundle_row.get("annotation_sha256") != annotation_ref.get("sha256"):
        _fail(f"{location} annotation bundle hash mismatch")
    if annotation.get("schema_version") != (
        "tauvision-complementary-inplace-document-v2"
    ):
        _fail(f"{location} annotation schema mismatch")
    if annotation.get("setting_id") != SETTING_ID:
        _fail(f"{location} annotation setting mismatch")
    if annotation.get("status") != "independent_review_passed":
        _fail(f"{location} annotation has not passed independent review")
    if annotation.get("document_id") != document_id:
        _fail(f"{location} annotation document ID mismatch")
    source_shard = _string(annotation.get("shard"), f"{location}.annotation.shard")
    if entry.get("shard") != source_shard or bundle_row.get("shard") != source_shard:
        _fail(f"{location} shard bindings differ")

    source = _object(annotation.get("source"), f"{location}.source")
    worklist_source = _object(worklist_row.get("source"), f"{location}.worklist.source")
    if source.get("path") != worklist_source.get("path"):
        _fail(f"{location} source paths differ")
    source_path = _resolve_contained(
        root, _string(source.get("path"), f"{location}.source.path"), "source"
    )
    source_payload, source_record = _load_json_bytes(source_path, f"{location}.source")
    source_file_hash = _sha256_bytes(source_payload)
    if source_file_hash != _sha256(source.get("file_sha256"), "source.file_sha256"):
        _fail(f"{location} source file hash mismatch")
    if source_file_hash != worklist_source.get("file_sha256"):
        _fail(f"{location} worklist source hash mismatch")
    if source_record.get("id") != document_id:
        _fail(f"{location} source document ID mismatch")
    title = _string(source_record.get("title"), f"{location}.source.title")
    content = _string(
        source_record.get("content"), f"{location}.source.content", allow_empty=True
    )
    if title != source.get("title") or _sha256_text(title) != source.get(
        "title_sha256"
    ):
        _fail(f"{location} source title binding mismatch")
    if worklist_row.get("title") != title:
        _fail(f"{location} worklist title mismatch")
    if _sha256_text(content) != source.get("content_sha256"):
        _fail(f"{location} source content hash mismatch")
    if source.get("content_sha256") != entry.get("source_content_sha256"):
        _fail(f"{location} manifest source content hash mismatch")
    if source.get("content_sha256") != bundle_row.get("source_content_sha256"):
        _fail(f"{location} bundle source content hash mismatch")
    if source.get("content_char_count") != len(content):
        _fail(f"{location} source character count mismatch")
    if source.get("content_byte_count") != len(content.encode("utf-8")):
        _fail(f"{location} source byte count mismatch")

    block = _object(annotation.get("block"), f"{location}.block")
    char_start = _nonnegative_int(block.get("char_start"), "block.char_start")
    char_end = _positive_int(block.get("char_end"), "block.char_end")
    if char_start >= char_end or char_end > len(content):
        _fail(f"{location} block character offsets are invalid")
    exact_block = content[char_start:char_end]
    if exact_block != block.get("exact_text"):
        _fail(f"{location} block text does not match character offsets")
    block_hash = _sha256_text(exact_block)
    if block_hash != block.get("exact_text_sha256"):
        _fail(f"{location} block text hash mismatch")
    if block_hash != entry.get("block_sha256"):
        _fail(f"{location} manifest block hash mismatch")
    if block_hash != bundle_row.get("block_sha256"):
        _fail(f"{location} bundle block hash mismatch")
    prefix = content[:char_start]
    suffix = content[char_end:]
    if _sha256_text(prefix) != block.get("prefix_sha256"):
        _fail(f"{location} prefix hash mismatch")
    if _sha256_text(suffix) != block.get("suffix_sha256"):
        _fail(f"{location} suffix hash mismatch")
    if (prefix == "") is not block.get("prefix_is_empty"):
        _fail(f"{location} prefix emptiness flag mismatch")
    if (suffix == "") is not block.get("suffix_is_empty"):
        _fail(f"{location} suffix emptiness flag mismatch")
    if _sha256_text(prefix + exact_block + suffix) != block.get(
        "reconstruction_sha256"
    ):
        _fail(f"{location} P+B+S reconstruction mismatch")
    byte_start = _nonnegative_int(block.get("byte_start"), "block.byte_start")
    byte_end = _positive_int(block.get("byte_end"), "block.byte_end")
    if byte_start != len(prefix.encode("utf-8")) or byte_end != len(
        (prefix + exact_block).encode("utf-8")
    ):
        _fail(f"{location} byte offsets do not match UTF-8 character slicing")
    expected_form = (
        "B"
        if not prefix and not suffix
        else "B_S"
        if not prefix
        else "P_B"
        if not suffix
        else "P_B_S"
    )
    if expected_form not in FORMS or block.get("decomposition_form") != expected_form:
        _fail(f"{location} decomposition form mismatch")
    if entry.get("decomposition_form") != expected_form:
        _fail(f"{location} manifest decomposition form mismatch")
    if bundle_row.get("decomposition_form") != expected_form:
        _fail(f"{location} bundle decomposition form mismatch")

    asset = _object(entry.get("asset"), f"{location}.asset")
    annotated_asset = _object(annotation.get("asset"), f"{location}.annotation.asset")
    for field in ("path", "sha256", "mime_type", "width", "height", "version"):
        if asset.get(field) != annotated_asset.get(field):
            _fail(f"{location} manifest/annotation asset field {field!r} differs")
    if asset.get("mime_type") != "image/png":
        _fail(f"{location} asset MIME type must be image/png")
    image_path = _resolve_contained(
        root, _string(asset.get("path"), f"{location}.asset.path"), "asset"
    )
    image_payload = image_path.read_bytes()
    image_hash = _sha256(asset.get("sha256"), f"{location}.asset.sha256")
    if _sha256_bytes(image_payload) != image_hash:
        _fail(f"{location} image hash mismatch")
    expected_dimensions = (
        _positive_int(asset.get("width"), f"{location}.asset.width"),
        _positive_int(asset.get("height"), f"{location}.asset.height"),
    )
    if _png_dimensions(image_payload, location) != expected_dimensions:
        _fail(f"{location} image dimensions mismatch")
    if bundle_row.get("image_path") != asset.get("path"):
        _fail(f"{location} bundle image path mismatch")
    if bundle_row.get("image_sha256") != image_hash:
        _fail(f"{location} bundle image hash mismatch")
    if (
        bundle_row.get("image_width") != expected_dimensions[0]
        or bundle_row.get("image_height") != expected_dimensions[1]
        or bundle_row.get("asset_version") != asset.get("version")
    ):
        _fail(f"{location} bundle image metadata mismatch")
    annotation_generation = _object(
        annotation.get("generation"), f"{location}.annotation.generation"
    )
    if (
        annotation_generation.get("tool") != "image_gen.imagegen"
        or annotation_generation.get("mode") != "built_in"
        or annotation_generation.get("programmatic_rendering") is not False
        or annotation_generation.get("programmatic_text_overlay") is not False
    ):
        _fail(f"{location} annotation generation contract mismatch")

    generation_ref = _object(entry.get("generation"), f"{location}.generation")
    _, generation = _bound_json(
        root,
        generation_ref.get("path"),
        generation_ref.get("sha256"),
        f"{location}.generation",
    )
    if bundle_row.get("generation_path") != generation_ref.get(
        "path"
    ) or bundle_row.get("generation_sha256") != generation_ref.get("sha256"):
        _fail(f"{location} bundle generation binding mismatch")
    if generation.get("schema_version") != (
        "tauvision-complementary-inplace-generation-v2"
    ):
        _fail(f"{location} generation schema mismatch")
    if generation.get("status") != "independent_review_passed":
        _fail(f"{location} generation is not independently accepted")
    if generation.get("document_id") != document_id:
        _fail(f"{location} generation document ID mismatch")
    generation_contract = _object(
        generation.get("generation_contract"),
        f"{location}.generation.generation_contract",
    )
    if (
        generation_contract.get("tool") != "image_gen.imagegen"
        or generation_contract.get("mode") != "built_in"
        or generation_contract.get("programmatic_rendering") is not False
        or generation_contract.get("programmatic_text_overlay") is not False
    ):
        _fail(f"{location} generation record contract mismatch")
    selected_asset = _object(
        generation.get("selected_asset"), f"{location}.generation.selected_asset"
    )
    for field in ("path", "sha256", "mime_type", "width", "height"):
        if selected_asset.get(field) != asset.get(field):
            _fail(f"{location} generation asset field {field!r} differs")
    if (
        selected_asset.get("attempt") != asset.get("attempt")
        or selected_asset.get("version") != asset.get("version")
        or generation_ref.get("selected_attempt") != asset.get("attempt")
    ):
        _fail(f"{location} generation asset version/attempt mismatch")
    generation_annotation = _object(
        generation.get("annotation"), f"{location}.generation.annotation"
    )
    if generation_annotation.get("block_sha256") != block_hash:
        _fail(f"{location} generation block binding mismatch")
    if generation_annotation.get("source_content_sha256") != _sha256_text(content):
        _fail(f"{location} generation source binding mismatch")

    review_ref = _object(entry.get("review"), f"{location}.review")
    _, review = _bound_json(
        root,
        review_ref.get("path"),
        review_ref.get("sha256"),
        f"{location}.review",
    )
    if bundle_row.get("review_path") != review_ref.get("path") or bundle_row.get(
        "review_sha256"
    ) != review_ref.get("sha256"):
        _fail(f"{location} bundle review binding mismatch")
    if review.get("schema_version") != (
        "tauvision-complementary-inplace-independent-review-v2"
    ):
        _fail(f"{location} review schema mismatch")
    if review.get("overall_verdict") != "PASS":
        _fail(f"{location} independent review did not pass")
    if review.get("document_id") != document_id:
        _fail(f"{location} review document ID mismatch")
    reviewer = _object(review.get("reviewer"), f"{location}.review.reviewer")
    if reviewer.get("independent") is not True:
        _fail(f"{location} review is not independent")
    if reviewer.get("bitmap_inspected_at_original_detail") is not True:
        _fail(f"{location} bitmap was not reviewed at original detail")
    reviewer_shard = _string(reviewer.get("shard"), f"{location}.review.reviewer.shard")
    if (
        review.get("source_shard") != source_shard
        or reviewer_shard == source_shard
        or review_ref.get("reviewer_shard") != reviewer_shard
    ):
        _fail(f"{location} reviewer is not cross-shard independent")
    review_bindings = _object(review.get("bindings"), f"{location}.review.bindings")
    if (
        review_bindings.get("selected_asset_path") != asset.get("path")
        or review_bindings.get("selected_asset_sha256") != image_hash
        or review_bindings.get("block_sha256") != block_hash
        or review_bindings.get("selected_asset_version") != asset.get("version")
        or review_bindings.get("selected_attempt") != asset.get("attempt")
    ):
        _fail(f"{location} review bindings mismatch")
    criteria = _list(review.get("criteria"), f"{location}.review.criteria")
    if not criteria or any(
        _object(row, f"{location}.review.criteria").get("verdict") != "PASS"
        for row in criteria
    ):
        _fail(f"{location} review contains a non-PASS criterion")
    annotation_review = _object(
        annotation.get("review"), f"{location}.annotation.review"
    )
    if (
        annotation_review.get("status") != "PASS"
        or annotation_review.get("reviewer_shard") != reviewer_shard
        or annotation_review.get("selected_asset_sha256") != image_hash
        or annotation_review.get("round") != review.get("round")
    ):
        _fail(f"{location} annotation review summary mismatch")

    # Keep the path read above alive in diagnostics and guard accidental swaps.
    if annotation_path.name != Path(str(annotation_ref.get("path"))).name:
        _fail(f"{location} resolved annotation filename mismatch")
    return RuntimeDocument(
        document_id=document_id,
        document_key=document_key,
        title=title,
        source_content=content,
        prefix=prefix,
        block=exact_block,
        suffix=suffix,
        char_start=char_start,
        char_end=char_end,
        decomposition_form=expected_form,
        image_payload=image_payload,
        image_sha256=image_hash,
        image_width=expected_dimensions[0],
        image_height=expected_dimensions[1],
        image_path=image_path,
    )


def load_task_runtime(
    *,
    task: Task,
    domain_policy: str,
    arm: str | None = None,
    manifest_path: Path | None = None,
    asset_root: Path | None = None,
) -> RuntimeTask:
    """Validate and materialize one task from the frozen 60-document bundle."""
    selected_arm = resolve_arm(arm)
    resolved_manifest, root = resolve_manifest_path(
        manifest_path, asset_root=asset_root
    )
    manifest_payload, manifest = _load_json_bytes(resolved_manifest, "runtime manifest")
    if _sha256_bytes(manifest_payload) != EXPECTED_MANIFEST_SHA256:
        _fail("runtime manifest hash differs from the frozen v2 release")
    manifest_by_id, bundle_by_id, worklist_by_id, selection_by_task = (
        _validate_global_bundle(manifest, root)
    )
    if task.id not in selection_by_task:
        _fail(f"task {task.id!r} is outside the frozen 5-of-5 cohort")
    task_row = selection_by_task[task.id]
    required_documents = task.required_documents or []
    selected_documents = _list(
        task_row.get("required_documents"), f"selection task {task.id}.documents"
    )
    if required_documents != selected_documents:
        _fail(f"task {task.id} required-document order differs from the frozen cohort")
    if not required_documents:
        _fail(f"task {task.id} has no required documents")

    task_source_path = _resolve_contained(
        root,
        _string(task_row.get("task_source_path"), "selection task source path"),
        "selection task source",
    )
    if _sha256_bytes(task_source_path.read_bytes()) != _sha256(
        task_row.get("task_source_sha256"), "selection task source hash"
    ):
        _fail(f"task {task.id} source file hash mismatch")

    manifest_docs_for_task = {
        document_id
        for document_id, entry in manifest_by_id.items()
        if task.id in _list(entry.get("task_ids"), "manifest entry.task_ids")
    }
    if manifest_docs_for_task != set(required_documents):
        _fail(f"task {task.id} manifest document set differs from task source")

    documents: list[RuntimeDocument] = []
    for document_id in required_documents:
        if (
            document_id not in manifest_by_id
            or document_id not in bundle_by_id
            or document_id not in worklist_by_id
        ):
            _fail(f"task {task.id} document {document_id!r} is missing from the bundle")
        documents.append(
            _validate_source_and_block(
                task_id=task.id,
                entry=manifest_by_id[document_id],
                bundle_row=bundle_by_id[document_id],
                worklist_row=worklist_by_id[document_id],
                root=root,
            )
        )

    from tau2.domains.banking_knowledge.retrieval import (
        PROMPTS_DIR,
        load_prompt_template,
    )

    required_text = "\n\n---\n\n".join(
        f"## {document.title}\n\n{document.source_content}" for document in documents
    )
    expected_policy = load_prompt_template(
        PROMPTS_DIR / "required_docs.md", knowledge_base=None
    ).replace("{{required_documents}}", required_text)
    if domain_policy != expected_policy:
        _fail(
            f"task {task.id} policy is not the byte-identical Golden Retrieval policy"
        )
    return RuntimeTask(
        arm=selected_arm,
        task_id=task.id,
        domain_policy=domain_policy,
        documents=tuple(documents),
        manifest_path=resolved_manifest,
        asset_root=root,
    )
