"""Fail-closed runtime loader for the source-native v4 image release.

The v4 task cohort is deliberately read from the frozen asset manifest.  No
task IDs or image paths are inferred from an earlier release.  Until the final
218-image pixel gate passes and :data:`EXPECTED_MANIFEST_SHA256` is pinned,
every public loader fails before an agent can make a model request.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tau2.data_model.tasks import Task


MANIFEST_ENV = "TAU2_COMPLEMENTARY_INPLACE_V4_MANIFEST"
ASSET_ROOT_ENV = "TAU2_COMPLEMENTARY_INPLACE_V4_ASSET_ROOT"
ARM_ENV = "TAU2_COMPLEMENTARY_INPLACE_V4_ARM"
SCHEMA_VERSION = "tauvision-complementary-inplace-asset-manifest-v4"
SETTING_ID = "complementary_inplace_system_v4"
GENERATION_RECORD_V4_SCHEMA = (
    "tauvision-complementary-inplace-source-native-imagegen-record-v4"
)
RETRY_GENERATION_RECORD_V5_SCHEMA = (
    "tauvision-complementary-inplace-source-native-retry-imagegen-record-v5"
)
RETRY_CANDIDATE_PROVENANCE_SCHEMA = (
    "tauvision-complementary-inplace-source-native-retry-candidate-provenance-v1"
)
ARMS = ("inplace_text", "inplace_image")
EXPECTED_TASK_COUNT = 97
EXPECTED_DOCUMENT_COUNT = 218
EXPECTED_DOCUMENT_REFERENCE_COUNT = 959
EXPECTED_ASSET_ORIGINS = {"fresh_builtin_imagegen": 218}
EXPECTED_MANIFEST_SHA256 = "8f80c74c8a79170cc5300d36bfbbc9bc89d032e464d0348668fb43c8c8797fd7"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_DEFAULT_MANIFEST = Path(
    "tauvision/data/banking_explorer/complementary_inplace_system_v4/asset_manifest.json"
)


class ComplementaryInplaceSystemV4Error(ValueError):
    """A source-native v4 runtime invariant failed."""


@dataclass(frozen=True)
class RuntimeDocument:
    """One exact document block and its independently reviewed replacement."""

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
    """The frozen, ordered runtime projection for one task and arm."""

    arm: str
    task_id: str
    domain_policy: str
    documents: tuple[RuntimeDocument, ...]
    manifest_path: Path
    asset_root: Path
    manifest_sha256: str


@dataclass(frozen=True)
class ManifestMetadata:
    """Model-free cohort metadata read from the hash-pinned manifest."""

    path: Path
    asset_root: Path
    sha256: str
    task_ids: tuple[str, ...]
    document_ids: tuple[str, ...]
    document_reference_count: int


def _fail(message: str) -> None:
    raise ComplementaryInplaceSystemV4Error(message)


def _require(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _text_sha(value: str) -> str:
    return _sha(value.encode("utf-8"))


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return _sha(payload)


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


def expected_manifest_sha256() -> str:
    """Return the release pin, failing while the final image gate is pending."""

    if _SHA256_RE.fullmatch(EXPECTED_MANIFEST_SHA256) is None:
        _fail(
            "v4 runtime is not released: EXPECTED_MANIFEST_SHA256 must be pinned "
            "after 218/218 independent pixel PASS"
        )
    return EXPECTED_MANIFEST_SHA256


def _infer_root(origin: Path | None = None) -> Path:
    starts = (origin, Path.cwd(), Path(__file__).resolve().parent)
    for start in starts:
        if start is None:
            continue
        for candidate in (start, *start.parents):
            if (candidate / "tau2-bench").is_dir() and (
                candidate / "tauvision"
            ).is_dir():
                return candidate.resolve()
    _fail(f"{ASSET_ROOT_ENV} is unset and the workspace root cannot be inferred")


def resolve_asset_root(path: Path | None = None) -> Path:
    """Resolve and validate the shared Tau-Vision workspace root."""

    configured = os.environ.get(ASSET_ROOT_ENV)
    raw = path or (Path(configured) if configured else None) or _infer_root()
    try:
        root = raw.expanduser().resolve(strict=True)
    except OSError as exc:
        _fail(f"cannot resolve v4 asset root {raw}: {exc}")
    _require(root.is_dir(), f"v4 asset root is not a directory: {root}")
    _require(
        (root / "tau2-bench").is_dir() and (root / "tauvision").is_dir(),
        f"invalid Tau-Vision asset root: {root}",
    )
    return root


def _contained(root: Path, raw: Any, location: str) -> Path:
    if not isinstance(raw, str) or not raw:
        _fail(f"{location} must be a nonempty workspace-relative path")
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        _fail(f"{location} is not a contained workspace-relative path")
    try:
        path = (root / relative).resolve(strict=True)
        path.relative_to(root)
    except (OSError, ValueError) as exc:
        _fail(f"{location} escapes or is missing from the asset root: {exc}")
    _require(path.is_file(), f"{location} is not a regular file")
    return path


def _binding(
    root: Path, value: Any, location: str
) -> tuple[Path, bytes, dict[str, Any]]:
    ref = _object(value, location)
    path = _contained(root, ref.get("path"), location)
    payload, parsed = _load(path, location)
    _require(_sha(payload) == ref.get("sha256"), f"{location} hash mismatch")
    return path, payload, parsed


def resolve_manifest_path(
    path: Path | None = None, *, asset_root: Path | None = None
) -> tuple[Path, Path]:
    """Resolve the runtime manifest without weakening its hash pin."""

    root = resolve_asset_root(asset_root)
    configured = os.environ.get(MANIFEST_ENV)
    raw = path or (Path(configured) if configured else _DEFAULT_MANIFEST)
    candidate = raw if raw.is_absolute() else root / raw
    try:
        resolved = candidate.expanduser().resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        _fail(f"runtime manifest escapes or is missing from the asset root: {exc}")
    _require(resolved.is_file(), "runtime manifest is not a regular file")
    return resolved, root


def resolve_arm(arm: str | None = None) -> str:
    """Resolve one of the two matched system-message conditions."""

    value = arm or os.environ.get(ARM_ENV)
    if value not in ARMS:
        _fail(f"{ARM_ENV} must be one of {list(ARMS)}, got {value!r}")
    return value


def _index(rows: list[Any], key: str, location: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(rows):
        row = _object(raw, f"{location}[{index}]")
        identifier = row.get(key)
        if not isinstance(identifier, str) or not identifier or identifier in result:
            _fail(f"{location} has an invalid or duplicate {key}")
        result[identifier] = row
    return result


def _validate_manifest(
    manifest: dict[str, Any], root: Path
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    _require(
        manifest.get("schema_version") == SCHEMA_VERSION, "manifest schema mismatch"
    )
    _require(manifest.get("setting_id") == SETTING_ID, "manifest setting mismatch")
    _require(manifest.get("status") == "frozen_run_eligible", "manifest is not frozen")
    _require(manifest.get("run_eligible") is True, "manifest is not run eligible")

    counts = _object(manifest.get("counts"), "manifest.counts")
    _require(counts.get("tasks") == EXPECTED_TASK_COUNT, "manifest task count mismatch")
    _require(
        counts.get("documents") == EXPECTED_DOCUMENT_COUNT,
        "manifest document count mismatch",
    )
    _require(
        counts.get("document_references") == EXPECTED_DOCUMENT_REFERENCE_COUNT,
        "manifest document-reference count mismatch",
    )
    _require(
        counts.get("asset_origins") == EXPECTED_ASSET_ORIGINS,
        "manifest does not contain 218 fresh ImageGen assets",
    )
    _require(counts.get("pixel_pass") == 218, "manifest pixel PASS count mismatch")
    _require(counts.get("pixel_fail") == 0, "manifest contains failed pixels")
    _require(counts.get("reused_assets") == 0, "manifest contains reused assets")

    tasks = _index(
        _rows(manifest.get("tasks"), "manifest.tasks"), "task_id", "manifest.tasks"
    )
    entries = _index(
        _rows(manifest.get("entries"), "manifest.entries"),
        "document_id",
        "manifest.entries",
    )
    _require(len(tasks) == EXPECTED_TASK_COUNT, "manifest task closure mismatch")
    _require(len(entries) == EXPECTED_DOCUMENT_COUNT, "manifest entry closure mismatch")

    bindings = _object(manifest.get("bindings"), "manifest.bindings")
    _, _, population = _binding(root, bindings.get("population"), "population")
    _require(
        population.get("schema_version")
        == "tauvision-complementary-inplace-population-v4",
        "population schema mismatch",
    )
    _require(population.get("setting_id") == SETTING_ID, "population setting mismatch")
    population_counts = _object(population.get("counts"), "population.counts")
    _require(
        population_counts.get("tasks") == EXPECTED_TASK_COUNT
        and population_counts.get("unique_required_documents")
        == EXPECTED_DOCUMENT_COUNT
        and population_counts.get("document_references")
        == EXPECTED_DOCUMENT_REFERENCE_COUNT,
        "population closure mismatch",
    )
    for label in (
        "annotation_bundle",
        "annotation_worklist",
        "source_native_manifest",
    ):
        _binding(root, bindings.get(label), label)
    freezer_ref = _object(
        bindings.get("asset_manifest_freezer"), "asset_manifest_freezer"
    )
    freezer_path = _contained(root, freezer_ref.get("path"), "asset_manifest_freezer")
    _require(
        _sha(freezer_path.read_bytes()) == freezer_ref.get("sha256"),
        "asset manifest freezer hash drift",
    )
    _, _, annotation_report = _binding(
        root,
        bindings.get("annotation_validation_report"),
        "annotation_validation_report",
    )
    _require(
        annotation_report.get("status") == "PASS",
        "annotation validation report is not PASS",
    )
    _, _, imagegen = _binding(
        root, bindings.get("imagegen_worklist"), "imagegen_worklist"
    )
    _require(
        imagegen.get("schema_version")
        == "tauvision-complementary-inplace-source-native-imagegen-worklist-v4",
        "ImageGen worklist schema mismatch",
    )
    imagegen_counts = _object(imagegen.get("counts"), "imagegen_worklist.counts")
    _require(
        imagegen_counts.get("documents") == 218
        and imagegen_counts.get("fresh_builtin_imagegen_required") == 218
        and imagegen_counts.get("imagegen_calls_required") == 218
        and imagegen_counts.get("final_reuse_allowed") == 0
        and imagegen_counts.get("actions") == {"generate_with_builtin_imagegen": 218},
        "ImageGen worklist is not fresh-218/reuse-0",
    )
    imagegen_items = _rows(imagegen.get("items"), "imagegen_worklist.items")
    _require(
        len(imagegen_items) == 218
        and all(
            isinstance(item, dict)
            and item.get("action") == "generate_with_builtin_imagegen"
            and item.get("reuse_allowed") is False
            for item in imagegen_items
        ),
        "ImageGen worklist contains a non-fresh route",
    )
    _, _, closure = _binding(
        root, bindings.get("pixel_review_closure"), "pixel_review_closure"
    )
    _require(
        closure.get("schema_version")
        == "tauvision-complementary-inplace-source-native-final-pixel-review-closure-v4",
        "pixel-review closure schema mismatch",
    )
    _require(
        closure.get("image_gate_passed") is True,
        "pixel-review closure did not pass",
    )
    closure_counts = _object(closure.get("counts"), "pixel_review_closure.counts")
    _require(
        closure_counts.get("total") == 218
        and closure_counts.get("pass") == 218
        and closure_counts.get("fail") == 0
        and closure_counts.get("new_attempts_required") == 0,
        "pixel-review closure is not 218 PASS / 0 FAIL",
    )

    population_tasks = _rows(population.get("tasks"), "population.tasks")
    population_documents = _rows(population.get("documents"), "population.documents")
    _require(
        [row.get("task_id") for row in population_tasks] == list(tasks),
        "manifest task order differs from the hash-bound population",
    )
    _require(
        [row.get("document_id") for row in population_documents] == list(entries),
        "manifest document order differs from the hash-bound population",
    )
    population_task_by_id = _index(population_tasks, "task_id", "population.tasks")
    population_document_by_id = _index(
        population_documents, "document_id", "population.documents"
    )

    references = 0
    referenced_documents: set[str] = set()
    for task_id, row in tasks.items():
        source = _object(row.get("source"), f"task {task_id}.source")
        population_task = population_task_by_id[task_id]
        _require(
            source
            == {
                "path": population_task.get("path"),
                "sha256": population_task.get("sha256"),
            },
            f"task {task_id} source binding differs from population",
        )
        task_path = _contained(root, source.get("path"), f"task {task_id}.source")
        _require(
            _sha(task_path.read_bytes()) == source.get("sha256"),
            f"task {task_id} source drift",
        )
        required = row.get("required_documents")
        _require(
            required == population_task.get("required_documents"),
            f"task {task_id} required documents differ from population",
        )
        _require(
            isinstance(required, list) and required, f"task {task_id} has no documents"
        )
        _require(
            len(required) == len(set(required)), f"task {task_id} repeats a document"
        )
        _require(
            all(value in entries for value in required),
            f"task {task_id} names an unknown document",
        )
        references += len(required)
        referenced_documents.update(required)
    _require(
        references == EXPECTED_DOCUMENT_REFERENCE_COUNT,
        "task reference closure mismatch",
    )
    _require(
        referenced_documents == set(entries), "some manifest documents are unreferenced"
    )

    asset_hashes: list[str] = []
    for document_id, entry in entries.items():
        population_document = population_document_by_id[document_id]
        _require(
            entry.get("document_key") == population_document.get("document_key")
            and entry.get("source_content_sha256")
            == population_document.get("content_sha256"),
            f"{document_id} identity/content differs from population",
        )
        task_ids = entry.get("task_ids")
        _require(
            isinstance(task_ids, list) and task_ids,
            f"{document_id} has no task bindings",
        )
        expected = [
            task_id
            for task_id, row in tasks.items()
            if document_id in row["required_documents"]
        ]
        _require(task_ids == expected, f"{document_id} task bindings/order mismatch")
        occurrence_ids = {
            occurrence.get("task_id")
            for occurrence in _rows(
                population_document.get("occurrences"),
                f"population document {document_id}.occurrences",
            )
            if isinstance(occurrence, dict)
        }
        _require(
            set(task_ids) == occurrence_ids,
            f"{document_id} task bindings differ from population occurrences",
        )
        asset = _object(entry.get("asset"), f"{document_id}.asset")
        _require(
            asset.get("origin") == "fresh_builtin_imagegen",
            f"{document_id} is not a fresh ImageGen asset",
        )
        digest = asset.get("sha256")
        _require(
            isinstance(digest, str) and _SHA256_RE.fullmatch(digest) is not None,
            f"{document_id} has an invalid asset hash",
        )
        asset_hashes.append(digest)
    _require(
        len(set(asset_hashes)) == EXPECTED_DOCUMENT_COUNT,
        "one PNG is reused across documents",
    )
    return tasks, entries


@lru_cache(maxsize=4)
def _load_resolved_pinned_manifest(
    path_text: str, root_text: str, expected_sha: str
) -> tuple[Path, Path, str, dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Validate one immutable release once per process.

    Agent construction still re-hashes every task's source, annotation,
    generation record, review report, and PNG.  This cache only prevents the
    same large global bundles from being reparsed once per task-filter call.
    Post-run validation starts a fresh process and repeats the full closure.
    """

    path = Path(path_text)
    root = Path(root_text)
    payload, manifest = _load(path, "runtime manifest")
    actual_sha = _sha(payload)
    _require(
        actual_sha == expected_sha,
        "runtime manifest hash differs from the frozen v4 release",
    )
    tasks, entries = _validate_manifest(manifest, root)
    return path, root, actual_sha, tasks, entries


def _load_pinned_manifest(
    manifest_path: Path | None = None, asset_root: Path | None = None
) -> tuple[Path, Path, str, dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    expected_sha = expected_manifest_sha256()
    path, root = resolve_manifest_path(manifest_path, asset_root=asset_root)
    return _load_resolved_pinned_manifest(str(path), str(root), expected_sha)


def load_manifest_metadata(
    manifest_path: Path | None = None, asset_root: Path | None = None
) -> ManifestMetadata:
    """Load the hash-pinned 97/218/959 cohort without reading image bytes."""

    path, root, digest, tasks, entries = _load_pinned_manifest(
        manifest_path, asset_root
    )
    references = sum(len(row["required_documents"]) for row in tasks.values())
    return ManifestMetadata(
        path=path,
        asset_root=root,
        sha256=digest,
        task_ids=tuple(tasks),
        document_ids=tuple(entries),
        document_reference_count=references,
    )


def is_frozen_task(task: Task) -> bool:
    """Return membership in the manifest-defined cohort, failing if unfrozen."""

    metadata = load_manifest_metadata()
    return task.id in metadata.task_ids


def _load_document(
    *, task_id: str, entry: dict[str, Any], root: Path
) -> RuntimeDocument:
    document_id = entry["document_id"]
    _require(task_id in entry["task_ids"], f"{document_id} lacks task binding")
    _, _, annotation = _binding(
        root, entry.get("annotation"), f"{document_id}.annotation"
    )
    _require(
        annotation.get("schema_version")
        == "tauvision-complementary-inplace-document-v4",
        f"{document_id} annotation schema mismatch",
    )
    _require(
        annotation.get("setting_id") == SETTING_ID,
        f"{document_id} annotation setting mismatch",
    )
    _require(
        annotation.get("document_id") == document_id,
        f"{document_id} annotation identity mismatch",
    )
    _require(
        annotation.get("document_key") == entry.get("document_key"),
        f"{document_id} key mismatch",
    )

    source_ref = _object(annotation.get("source"), f"{document_id}.source")
    source_path = _contained(root, source_ref.get("path"), f"{document_id}.source")
    source_payload, source = _load(source_path, f"{document_id}.source")
    _require(
        _sha(source_payload) == source_ref.get("file_sha256"),
        f"{document_id} source file drift",
    )
    content = source.get("content")
    title = source.get("title")
    _require(
        isinstance(content, str) and isinstance(title, str),
        f"{document_id} source fields malformed",
    )
    _require(
        _text_sha(content) == source_ref.get("content_sha256"),
        f"{document_id} content hash mismatch",
    )
    _require(
        _text_sha(title) == source_ref.get("title_sha256"),
        f"{document_id} title hash mismatch",
    )
    _require(
        _text_sha(content) == entry.get("source_content_sha256"),
        f"{document_id} manifest content mismatch",
    )

    block = _object(annotation.get("block"), f"{document_id}.block")
    start, end = block.get("char_start"), block.get("char_end")
    _require(
        type(start) is int and type(end) is int,
        f"{document_id} block offsets malformed",
    )
    _require(0 <= start < end <= len(content), f"{document_id} block offsets invalid")
    prefix, selected, suffix = content[:start], content[start:end], content[end:]
    form = (
        "B"
        if not prefix and not suffix
        else "B_S"
        if not prefix
        else "P_B"
        if not suffix
        else "P_B_S"
    )
    _require(selected == block.get("exact_text"), f"{document_id} exact block moved")
    _require(
        _text_sha(selected) == block.get("exact_text_sha256"),
        f"{document_id} block hash mismatch",
    )
    _require(
        _text_sha(selected) == entry.get("block_sha256"),
        f"{document_id} manifest block mismatch",
    )
    _require(
        form == block.get("decomposition_form") == entry.get("decomposition_form"),
        f"{document_id} decomposition mismatch",
    )
    _require(
        prefix + selected + suffix == content,
        f"{document_id} P+B+S reconstruction failed",
    )

    asset = _object(entry.get("asset"), f"{document_id}.asset")
    image_path = _contained(root, asset.get("path"), f"{document_id}.asset")
    image = image_path.read_bytes()
    _require(_sha(image) == asset.get("sha256"), f"{document_id} image hash mismatch")
    _require(
        len(image) >= 24 and image[:8] == b"\x89PNG\r\n\x1a\n",
        f"{document_id} image is not PNG",
    )
    _require(image[12:16] == b"IHDR", f"{document_id} image lacks canonical IHDR")
    width, height = struct.unpack(">II", image[16:24])
    _require(
        (width, height) == (asset.get("width"), asset.get("height")),
        f"{document_id} image dimensions mismatch",
    )

    _, _, generation = _binding(
        root, entry.get("generation_record"), f"{document_id}.generation_record"
    )
    attempt = asset.get("attempt")
    _require(type(attempt) is int and attempt >= 1, f"{document_id} invalid generation attempt")
    _require(generation.get("attempt") == attempt, f"{document_id} generation/asset attempt mismatch")
    if attempt == 1:
        _require(
            generation.get("schema_version") == GENERATION_RECORD_V4_SCHEMA,
            f"{document_id} attempt-1 generation record schema mismatch",
        )
        _require(
            "retry_after_failure" not in generation
            and "actual_generation_provenance" not in generation,
            f"{document_id} attempt-1 record contains retry provenance",
        )
    else:
        _require(
            generation.get("schema_version") == RETRY_GENERATION_RECORD_V5_SCHEMA,
            f"{document_id} retry generation record schema mismatch",
        )
        retry = _object(
            generation.get("retry_after_failure"),
            f"{document_id}.generation_record.retry_after_failure",
        )
        _require(
            retry.get("required_generation_attempt") == attempt
            and retry.get("prior_generation_attempt") == attempt - 1,
            f"{document_id} retry attempt provenance mismatch",
        )
        provenance = _object(
            generation.get("actual_generation_provenance"),
            f"{document_id}.generation_record.actual_generation_provenance",
        )
        _require(
            provenance.get("schema_version") == RETRY_CANDIDATE_PROVENANCE_SCHEMA,
            f"{document_id} retry candidate provenance schema mismatch",
        )
        details = _object(
            generation.get("generation"), f"{document_id}.generation_record.generation"
        )
        actual_prompt = details.get("prompt")
        _require(
            isinstance(actual_prompt, str)
            and _text_sha(actual_prompt) == details.get("prompt_sha256"),
            f"{document_id} retry actual prompt hash mismatch",
        )
        lineage = _object(
            provenance.get("prompt_lineage"),
            f"{document_id}.generation_record.prompt_lineage",
        )
        _require(
            lineage.get("actual_prompt_sha256") == details.get("prompt_sha256"),
            f"{document_id} retry prompt-lineage hash mismatch",
        )
        repair_prompt = _object(
            retry.get("repair_prompt"), f"{document_id}.retry_after_failure.repair_prompt"
        )
        round1_text = repair_prompt.get("text")
        _require(
            isinstance(round1_text, str)
            and _text_sha(round1_text) == repair_prompt.get("sha256")
            and lineage.get("round1_repair_prompt_sha256") == repair_prompt.get("sha256")
            and actual_prompt.count(round1_text) == 1,
            f"{document_id} immutable repair prompt is not uniquely embedded",
        )
        candidate = _object(
            provenance.get("candidate"), f"{document_id}.generation_record.candidate"
        )
        source_asset = _object(
            candidate.get("source_asset"), f"{document_id}.generation_record.source_asset"
        )
        _require(
            source_asset.get("sha256") == asset.get("sha256")
            and source_asset.get("width") == asset.get("width")
            and source_asset.get("height") == asset.get("height"),
            f"{document_id} selected candidate differs from runtime asset",
        )
        qc = _object(
            provenance.get("selection_qc"), f"{document_id}.generation_record.selection_qc"
        )
        _require(qc.get("decision") == "PASS", f"{document_id} staging selection QC is not PASS")
        frozen = _object(
            provenance.get("frozen_selection"),
            f"{document_id}.generation_record.frozen_selection",
        )
        frozen_item = _object(
            frozen.get("item_snapshot"), f"{document_id}.generation_record.frozen_selection.item"
        )
        _require(
            _canonical_sha(frozen_item) == frozen.get("item_canonical_sha256")
            and frozen_item.get("document_id") == document_id
            and frozen_item.get("formal_generation_attempt_to_register") == attempt
            and frozen_item.get("generator_id") == generation.get("generator_id")
            and frozen_item.get("actual_generation_prompt", {}).get("sha256")
            == details.get("prompt_sha256")
            and frozen_item.get("asset", {}).get("sha256") == asset.get("sha256")
            and frozen_item.get("selected_qc_decision") == "PASS",
            f"{document_id} embedded frozen selection does not close over retry",
        )
    _require(
        generation.get("setting_id") == SETTING_ID
        and generation.get("document_id") == document_id,
        f"{document_id} generation record identity mismatch",
    )
    generated_asset = _object(
        generation.get("asset"), f"{document_id}.generation_record.asset"
    )
    for key in ("path", "sha256", "mime_type", "width", "height", "attempt"):
        _require(
            generated_asset.get(key) == asset.get(key),
            f"{document_id} generation/asset binding mismatch for {key}",
        )
    review_ref = _object(entry.get("pixel_review"), f"{document_id}.pixel_review")
    _, _, review_report = _binding(root, review_ref, f"{document_id}.pixel_review")
    _require(
        review_ref.get("decision") == "PASS",
        f"{document_id} pixel decision is not PASS",
    )
    review_items = [
        value
        for value in _rows(review_report.get("items"), f"{document_id}.review.items")
        if isinstance(value, dict) and value.get("document_id") == document_id
    ]
    _require(len(review_items) == 1, f"{document_id} pixel review item is not unique")
    review_item = review_items[0]
    _require(
        review_item.get("decision") == "PASS",
        f"{document_id} bound review item is not PASS",
    )
    _require(
        _canonical_sha(review_item) == review_ref.get("item_canonical_sha256"),
        f"{document_id} pixel review item hash mismatch",
    )
    reviewed_asset = _object(
        _object(review_item.get("bindings"), f"{document_id}.review.bindings").get(
            "asset"
        ),
        f"{document_id}.review.bindings.asset",
    )
    _require(
        reviewed_asset.get("path") == asset.get("path")
        and reviewed_asset.get("sha256") == asset.get("sha256"),
        f"{document_id} pixel review binds a different asset",
    )
    return RuntimeDocument(
        document_id=document_id,
        document_key=entry["document_key"],
        title=title,
        source_content=content,
        prefix=prefix,
        block=selected,
        suffix=suffix,
        char_start=start,
        char_end=end,
        decomposition_form=form,
        image_payload=image,
        image_sha256=asset["sha256"],
        image_width=width,
        image_height=height,
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
    """Load one task only after validating the complete frozen release."""

    selected_arm = resolve_arm(arm)
    path, root, manifest_sha, tasks, entries = _load_pinned_manifest(
        manifest_path, asset_root
    )
    if task.id not in tasks:
        _fail(f"task {task.id!r} is outside the frozen v4 cohort")
    task_row = tasks[task.id]
    required = task.required_documents or []
    _require(
        required == task_row["required_documents"],
        f"task {task.id} required-document order drift",
    )
    documents = tuple(
        _load_document(task_id=task.id, entry=entries[document_id], root=root)
        for document_id in required
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
    _require(
        domain_policy == expected_policy,
        f"task {task.id} policy is not byte-identical Golden Retrieval text",
    )
    return RuntimeTask(
        arm=selected_arm,
        task_id=task.id,
        domain_policy=domain_policy,
        documents=documents,
        manifest_path=path,
        asset_root=root,
        manifest_sha256=manifest_sha,
    )
