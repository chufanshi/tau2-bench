"""Frozen runtime contract for the selected complementary experiment.

The runtime dataset is produced outside tau2 by Tau-Vision's versioned
builder.  This module is deliberately strict: task selection, policy text,
semantic allocation, document order, and bitmap bytes must all match the
frozen bundle before an agent can be constructed.

``partitioned_text`` and ``complementary`` use the exact same ``matched_ft``
system policy.  Their private document messages differ only in whether the
frozen ``F_I`` projection is supplied as text or as reviewed PNG assets.
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
    from tau2.domains.banking_knowledge.data_model import KnowledgeBase


DATASET_ENV = "TAU2_SELECTED_COMPLEMENTARY_DATASET"
ASSET_ROOT_ENV = "TAU2_SELECTED_COMPLEMENTARY_ASSET_ROOT"
ARM_ENV = "TAU2_SELECTED_COMPLEMENTARY_ARM"

SCHEMA_VERSION = "tauvision-selected-complementary-runtime-v1"
SETTING_ID = "golden_selected_complementary_matched_v1"
IMAGEGEN_SCHEMA_VERSION = "tauvision-selected-complementary-imagegen-runtime-v1"
IMAGEGEN_SETTING_ID = "golden_selected_complementary_imagegen_matched_v1"
RUNTIME_IDENTITIES = {
    SCHEMA_VERSION: SETTING_ID,
    IMAGEGEN_SCHEMA_VERSION: IMAGEGEN_SETTING_ID,
}
EXPECTED_TASK_COUNT = 19
EXPECTED_DOCUMENT_REFERENCE_COUNT = 109
EXPECTED_UNIQUE_DOCUMENT_COUNT = 60
EXPECTED_VISIBLE_SOURCE_OCCURRENCE_COUNT = 231
ARMS = ("full_text", "partitioned_text", "complementary")
PRIMARY_ARMS = ("complementary", "partitioned_text")
IMAGEGEN_CONTRACT_SCHEMA = (
    "tauvision-selected-complementary-imagegen-runtime-contract-v1"
)
IMAGEGEN_CONTRACT_ID = "complementary_5of5_imagegen_runtime_contract_v1"
IMAGEGEN_REBIND_SCHEMA = (
    "tauvision-selected-complementary-imagegen-realization-rebind-v1"
)
IMAGEGEN_ASSET_ROOT_PREFIX = "docs/exp4_explorer/assets/complementary_5of5_imagegen_v1/"

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class SelectedComplementaryRuntimeError(ValueError):
    """A selected-complementary runtime invariant was not satisfied."""


@dataclass(frozen=True)
class RuntimeDocument:
    """Validated per-document content supplied by the matched agent."""

    document_id: str
    bridge_text: str
    fi_text: str
    image_payloads: tuple[bytes, ...]


@dataclass(frozen=True)
class RuntimeTask:
    """Validated task-specific runtime material."""

    arm: str
    task_id: str
    policy: str
    documents: tuple[RuntimeDocument, ...]
    dataset_path: Path
    asset_root: Path


def _fail(message: str) -> None:
    raise SelectedComplementaryRuntimeError(message)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _canonical_sha256(value: Any) -> str:
    return _sha256_text(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


def _object(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{location} must be an object")
    return value


def _list(value: Any, location: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(f"{location} must be a list")
    return value


def _string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"{location} must be a non-empty string")
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


def _unique_strings(value: Any, location: str, *, nonempty: bool = True) -> list[str]:
    raw = _list(value, location)
    result = [_string(item, f"{location}[{index}]") for index, item in enumerate(raw)]
    if nonempty and not result:
        _fail(f"{location} must not be empty")
    if len(result) != len(set(result)):
        _fail(f"{location} contains duplicates")
    return result


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(f"cannot read runtime dataset {path}: {exc}")
    return _object(value, "dataset")


def resolve_dataset_path(path: Path | None = None) -> Path:
    """Resolve the frozen runtime dataset without depending on the cwd."""
    if path is None:
        raw = os.environ.get(DATASET_ENV)
        if not raw:
            _fail(f"{DATASET_ENV} must point to a frozen runtime dataset")
        path = Path(raw)
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as exc:
        _fail(f"cannot resolve runtime dataset {path}: {exc}")
    if not resolved.is_file():
        _fail(f"runtime dataset is not a regular file: {resolved}")
    return resolved


def _infer_asset_root(dataset_path: Path) -> Path:
    for origin in (dataset_path.parent, Path.cwd(), Path(__file__).resolve().parent):
        for candidate in (origin, *origin.parents):
            if (candidate / "tau2-bench").is_dir() and (
                candidate / "tauvision"
            ).is_dir():
                return candidate.resolve()
    _fail(f"{ASSET_ROOT_ENV} is unset and the workspace root could not be inferred")


def resolve_asset_root(dataset_path: Path, path: Path | None = None) -> Path:
    """Resolve the immutable bitmap root used by the frozen bundle."""
    if path is None:
        raw = os.environ.get(ASSET_ROOT_ENV)
        path = Path(raw) if raw else _infer_asset_root(dataset_path)
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as exc:
        _fail(f"cannot resolve asset root {path}: {exc}")
    if not resolved.is_dir():
        _fail(f"asset root is not a directory: {resolved}")
    return resolved


def resolve_arm(arm: str | None = None) -> str:
    """Return the explicitly selected experimental arm."""
    value = arm or os.environ.get(ARM_ENV)
    if value not in ARMS:
        _fail(f"{ARM_ENV} must be one of {list(ARMS)}, got {value!r}")
    return value


def _validate_imagegen_dataset(
    dataset: dict[str, Any], indexed_tasks: dict[str, dict[str, Any]]
) -> None:
    """Require image-gen-only provenance and same-document consistency."""
    contract = _object(dataset.get("imagegen_contract"), "dataset.imagegen_contract")
    declared_contract_hash = _sha256(
        contract.get("sha256"), "dataset.imagegen_contract.sha256"
    )
    contract_payload = {
        key: value for key, value in contract.items() if key != "sha256"
    }
    if _canonical_sha256(contract_payload) != declared_contract_hash:
        _fail("dataset.imagegen_contract hash mismatch")
    expected_scalars = {
        "schema_version": IMAGEGEN_CONTRACT_SCHEMA,
        "id": IMAGEGEN_CONTRACT_ID,
        "mode": "built_in_image_gen",
        "tool": "image_gen.imagegen",
        "programmatic_text_overlay": False,
        "asset_root_prefix": IMAGEGEN_ASSET_ROOT_PREFIX,
    }
    for field, expected in expected_scalars.items():
        if contract.get(field) != expected:
            _fail(f"dataset.imagegen_contract.{field} must be {expected!r}")

    def binding(value: Any, location: str, schema: str) -> dict[str, Any]:
        record = _object(value, location)
        _string(record.get("path"), f"{location}.path")
        _sha256(record.get("sha256"), f"{location}.sha256")
        if record.get("schema_version") != schema:
            _fail(f"{location}.schema_version must be {schema!r}")
        return record

    annotation_binding = binding(
        contract.get("annotation_bundle_binding"),
        "dataset.imagegen_contract.annotation_bundle_binding",
        "tauvision-complementary-annotation-correction-bundle-v2",
    )
    manifest_binding = binding(
        contract.get("asset_manifest_binding"),
        "dataset.imagegen_contract.asset_manifest_binding",
        "tauvision-complementary-imagegen-assets-v1",
    )
    review_binding = binding(
        contract.get("final_review_binding"),
        "dataset.imagegen_contract.final_review_binding",
        "tauvision-complementary-independent-review-v1",
    )
    _string(
        review_binding.get("review_id"),
        "dataset.imagegen_contract.final_review_binding.review_id",
    )
    provenance = _object(dataset.get("provenance"), "dataset.provenance")
    provenance_inputs = _object(provenance.get("inputs"), "dataset.provenance.inputs")
    for name, expected in (
        ("annotation", annotation_binding),
        ("asset_manifest", manifest_binding),
        ("review", review_binding),
    ):
        observed = _object(
            provenance_inputs.get(name), f"dataset.provenance.inputs.{name}"
        )
        for field in ("path", "sha256", "schema_version"):
            if observed.get(field) != expected.get(field):
                _fail(
                    f"dataset.imagegen_contract {name} binding differs from "
                    f"provenance.inputs.{name}.{field}"
                )

    shard_bindings = _list(
        contract.get("shard_manifest_bindings"),
        "dataset.imagegen_contract.shard_manifest_bindings",
    )
    if len(shard_bindings) != 3:
        _fail("dataset.imagegen_contract must bind exactly three shard manifests")
    shard_by_path: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(shard_bindings):
        location = f"dataset.imagegen_contract.shard_manifest_bindings[{index}]"
        row = binding(raw, location, "complementary-imagegen-shard-manifest-v1")
        path = row["path"]
        if path in shard_by_path or row.get("document_count") != 20:
            _fail(f"{location} is duplicate or does not bind 20 documents")
        shard_by_path[path] = row

    independent_bindings = _list(
        contract.get("independent_review_bindings"),
        "dataset.imagegen_contract.independent_review_bindings",
    )
    if len(independent_bindings) != 3:
        _fail("dataset.imagegen_contract must bind exactly three shard reviews")
    independent_paths: set[str] = set()
    reviewer_ids: set[str] = set()
    reviewed_shards: set[str] = set()
    for index, raw in enumerate(independent_bindings):
        location = f"dataset.imagegen_contract.independent_review_bindings[{index}]"
        row = binding(
            raw,
            location,
            "tauvision-complementary-imagegen-independent-shard-review-v1",
        )
        reviewer_id = _string(row.get("reviewer_id"), f"{location}.reviewer_id")
        reviewed_shard = _string(
            row.get("reviewed_shard"), f"{location}.reviewed_shard"
        )
        bound_shard = _object(
            row.get("bound_shard_manifest"), f"{location}.bound_shard_manifest"
        )
        shard_path = _string(
            bound_shard.get("path"), f"{location}.bound_shard_manifest.path"
        )
        shard = shard_by_path.get(shard_path)
        if shard is None or bound_shard.get("sha256") != shard.get("sha256"):
            _fail(f"{location} binds a shard outside the frozen three-shard set")
        if (
            row["path"] in independent_paths
            or reviewer_id in reviewer_ids
            or reviewed_shard in reviewed_shards
        ):
            _fail("dataset.imagegen_contract reuses a review, reviewer, or shard")
        independent_paths.add(row["path"])
        reviewer_ids.add(reviewer_id)
        reviewed_shards.add(reviewed_shard)

    counts = _object(contract.get("counts"), "dataset.imagegen_contract.counts")
    expected_counts = {
        "tasks": EXPECTED_TASK_COUNT,
        "document_references": EXPECTED_DOCUMENT_REFERENCE_COUNT,
        "unique_documents": EXPECTED_UNIQUE_DOCUMENT_COUNT,
        "assets": EXPECTED_UNIQUE_DOCUMENT_COUNT,
        "visible_source_occurrences": EXPECTED_VISIBLE_SOURCE_OCCURRENCE_COUNT,
        "initial_image_gen_calls": EXPECTED_UNIQUE_DOCUMENT_COUNT,
    }
    for field, expected in expected_counts.items():
        if counts.get(field) != expected:
            _fail(f"dataset.imagegen_contract.counts.{field} must be {expected}")
    image_gen_calls = _positive_int(
        counts.get("image_gen_calls"),
        "dataset.imagegen_contract.counts.image_gen_calls",
    )
    regeneration_calls = counts.get("regeneration_calls")
    if type(regeneration_calls) is not int or regeneration_calls < 0:
        _fail("dataset.imagegen_contract.counts.regeneration_calls is invalid")
    if image_gen_calls != EXPECTED_UNIQUE_DOCUMENT_COUNT + regeneration_calls:
        _fail("image_gen call counts do not equal initial calls plus regenerations")
    _sha256(
        contract.get("manifest_entries_sha256"),
        "dataset.imagegen_contract.manifest_entries_sha256",
    )
    _sha256(
        contract.get("review_rows_sha256"),
        "dataset.imagegen_contract.review_rows_sha256",
    )
    declared_manifest_entry_hashes_hash = _sha256(
        contract.get("manifest_entry_sha256s_sha256"),
        "dataset.imagegen_contract.manifest_entry_sha256s_sha256",
    )
    declared_review_row_hashes_hash = _sha256(
        contract.get("review_row_sha256s_sha256"),
        "dataset.imagegen_contract.review_row_sha256s_sha256",
    )
    declared_rebinds_hash = _sha256(
        contract.get("realization_rebinds_sha256"),
        "dataset.imagegen_contract.realization_rebinds_sha256",
    )
    unique_document_ids = _unique_strings(
        contract.get("unique_document_ids"),
        "dataset.imagegen_contract.unique_document_ids",
    )
    if len(unique_document_ids) != EXPECTED_UNIQUE_DOCUMENT_COUNT:
        _fail("dataset.imagegen_contract must order exactly 60 unique documents")

    document_references = 0
    unique_documents: dict[str, dict[str, Any]] = {}
    binding_hashes: dict[str, str] = {}
    for task_id, bundle in indexed_tasks.items():
        required = _unique_strings(
            bundle.get("required_documents"), f"task {task_id}.required_documents"
        )
        documents = _list(bundle.get("documents"), f"task {task_id}.documents")
        if len(documents) != len(required):
            _fail(f"task {task_id} document/reference counts differ")
        document_references += len(required)
        for ordinal, (raw_document, expected_id) in enumerate(
            zip(documents, required, strict=True)
        ):
            location = f"task {task_id}.documents[{ordinal}]"
            document = _object(raw_document, location)
            if (
                document.get("ordinal") != ordinal
                or document.get("document_id") != expected_id
            ):
                _fail(f"{location} order/identity differs from required_documents")
            binding_hash = _sha256(
                document.get("binding_sha256"), f"{location}.binding_sha256"
            )
            binding_payload = {
                key: value
                for key, value in document.items()
                if key not in {"ordinal", "binding_sha256"}
            }
            if _canonical_sha256(binding_payload) != binding_hash:
                _fail(f"{location} binding hash mismatch")
            if (
                expected_id in binding_hashes
                and binding_hashes[expected_id] != binding_hash
            ):
                _fail(f"document {expected_id} differs across selected tasks")
            binding_hashes[expected_id] = binding_hash
            unique_documents.setdefault(expected_id, document)
    if document_references != EXPECTED_DOCUMENT_REFERENCE_COUNT:
        _fail("runtime document-reference count is not the frozen 109")
    if set(unique_documents) != set(unique_document_ids):
        _fail("runtime unique-document closure differs from imagegen_contract")

    rebinds: list[dict[str, Any]] = []
    asset_paths: set[str] = set()
    manifest_entry_hashes: set[str] = set()
    review_row_hashes: set[str] = set()
    ordered_manifest_entry_hashes: list[str] = []
    ordered_review_row_hashes: list[str] = []
    visible_occurrence_count = 0
    for document_id in unique_document_ids:
        document = unique_documents[document_id]
        fi_projection = _object(
            document.get("fi_text_projection"),
            f"document {document_id}.fi_text_projection",
        )
        visible_occurrences = _list(
            fi_projection.get("visible_source_occurrences"),
            f"document {document_id}.fi_text_projection.visible_source_occurrences",
        )
        visible_occurrence_count += len(visible_occurrences)
        visible_hash = _sha256(
            fi_projection.get("visible_source_occurrences_sha256"),
            f"document {document_id}.fi_text_projection.visible_source_occurrences_sha256",
        )
        if _canonical_sha256(visible_occurrences) != visible_hash:
            _fail(f"document {document_id} visible-occurrence hash mismatch")
        assets = _list(document.get("assets"), f"document {document_id}.assets")
        if len(assets) != 1:
            _fail(f"document {document_id} must bind exactly one image-gen asset")
        asset = _object(assets[0], f"document {document_id}.assets[0]")
        asset_id = _string(asset.get("asset_id"), f"document {document_id}.asset_id")
        asset_path = _string(asset.get("path"), f"document {document_id}.asset.path")
        relative_asset_path = Path(asset_path)
        if (
            relative_asset_path.is_absolute()
            or ".." in relative_asset_path.parts
            or not asset_path.startswith(IMAGEGEN_ASSET_ROOT_PREFIX)
            or asset_path in asset_paths
        ):
            _fail(
                f"document {document_id} asset is outside/duplicate in image-gen root"
            )
        asset_paths.add(asset_path)
        asset_hash = _sha256(
            asset.get("sha256"), f"document {document_id}.asset.sha256"
        )
        generation = _object(
            asset.get("generation"), f"document {document_id}.asset.generation"
        )
        if (
            generation.get("mode") != "built_in_image_gen"
            or generation.get("tool") != "image_gen.imagegen"
            or generation.get("programmatic_text_overlay") is not False
        ):
            _fail(f"document {document_id} lacks image-gen-only asset provenance")
        manifest_entry_hash = _sha256(
            generation.get("manifest_entry_sha256"),
            f"document {document_id}.asset.generation.manifest_entry_sha256",
        )
        review_row_hash = _sha256(
            generation.get("review_row_sha256"),
            f"document {document_id}.asset.generation.review_row_sha256",
        )
        if (
            manifest_entry_hash in manifest_entry_hashes
            or review_row_hash in review_row_hashes
        ):
            _fail(
                "image-gen manifest/review row hashes are not one-to-one with documents"
            )
        manifest_entry_hashes.add(manifest_entry_hash)
        review_row_hashes.add(review_row_hash)
        ordered_manifest_entry_hashes.append(manifest_entry_hash)
        ordered_review_row_hashes.append(review_row_hash)
        planned_path = _string(
            asset.get("annotation_planned_path"),
            f"document {document_id}.asset.annotation_planned_path",
        )
        if (
            not planned_path.startswith(
                "docs/exp4_explorer/assets/complementary_5of5_v3/"
            )
            or planned_path == asset_path
        ):
            _fail(f"document {document_id} lacks the frozen old-to-imagegen rebind")
        rebind = _object(
            asset.get("realization_rebind"),
            f"document {document_id}.asset.realization_rebind",
        )
        if rebind.get("schema_version") != IMAGEGEN_REBIND_SCHEMA:
            _fail(f"document {document_id} realization rebind schema mismatch")
        rebind_hash = _sha256(
            rebind.get("sha256"),
            f"document {document_id}.asset.realization_rebind.sha256",
        )
        rebind_payload = {
            key: value for key, value in rebind.items() if key != "sha256"
        }
        if _canonical_sha256(rebind_payload) != rebind_hash:
            _fail(f"document {document_id} realization rebind hash mismatch")
        if (
            rebind.get("document_id") != document_id
            or rebind.get("asset_id") != asset_id
            or rebind.get("annotation_planned_path") != planned_path
            or rebind.get("manifest_entry_sha256") != manifest_entry_hash
            or rebind.get("review_row_sha256") != review_row_hash
        ):
            _fail(f"document {document_id} realization rebind identity mismatch")
        _sha256(
            rebind.get("annotation_image_projection_binding_sha256"),
            f"document {document_id}.rebind.annotation binding",
        )
        _sha256(
            rebind.get("visible_source_occurrences_sha256"),
            f"document {document_id}.rebind.annotation occurrence hash",
        )
        realized_asset = _object(
            rebind.get("realized_asset"),
            f"document {document_id}.rebind.realized_asset",
        )
        if (
            realized_asset.get("path") != asset_path
            or realized_asset.get("sha256") != asset_hash
            or realized_asset.get("width") != asset.get("width")
            or realized_asset.get("height") != asset.get("height")
        ):
            _fail(f"document {document_id} realization rebind asset mismatch")
        rebind_generation = _object(
            rebind.get("generation"), f"document {document_id}.rebind.generation"
        )
        if rebind_generation != {
            "mode": "built_in_image_gen",
            "tool": "image_gen.imagegen",
            "programmatic_text_overlay": False,
        }:
            _fail(f"document {document_id} realization generation mismatch")
        if (
            _sha256(
                asset.get("visible_source_occurrences_sha256"),
                f"document {document_id}.asset.visible_source_occurrences_sha256",
            )
            != visible_hash
        ):
            _fail(f"document {document_id} asset/F_I occurrence binding differs")
        rebinds.append(rebind)
    if visible_occurrence_count != EXPECTED_VISIBLE_SOURCE_OCCURRENCE_COUNT:
        _fail("runtime unique-document visible occurrence count is not 231")
    if _canonical_sha256(rebinds) != declared_rebinds_hash:
        _fail("runtime realization rebind closure/order hash mismatch")
    if (
        _canonical_sha256(ordered_manifest_entry_hashes)
        != declared_manifest_entry_hashes_hash
        or _canonical_sha256(ordered_review_row_hashes)
        != declared_review_row_hashes_hash
    ):
        _fail("runtime manifest/review row-hash closure differs from contract")


def _validate_dataset(dataset: dict[str, Any]) -> dict[str, dict[str, Any]]:
    schema_version = dataset.get("schema_version")
    if schema_version not in RUNTIME_IDENTITIES:
        _fail(
            "dataset.schema_version must be one of "
            f"{list(RUNTIME_IDENTITIES)!r}, got {schema_version!r}"
        )
    if dataset.get("status") != "frozen_run_eligible":
        _fail("dataset.status must be 'frozen_run_eligible'")

    setting = _object(dataset.get("setting"), "dataset.setting")
    expected_setting = {
        "id": RUNTIME_IDENTITIES[schema_version],
        "task_count": EXPECTED_TASK_COUNT,
        "arms": list(ARMS),
        "primary_contrast": list(PRIMARY_ARMS),
        "allocation": {
            "text_atoms": "F_T",
            "image_atoms": "F_I",
            "intersection": "empty",
            "union": "F",
        },
    }
    for field, expected in expected_setting.items():
        if setting.get(field) != expected:
            _fail(
                f"dataset.setting.{field} is {setting.get(field)!r}; "
                f"expected {expected!r}"
            )

    selection = _object(dataset.get("selection"), "dataset.selection")
    selection_hash = _sha256(
        selection.get("payload_sha256"), "dataset.selection.payload_sha256"
    )
    selection_payload = {
        key: value for key, value in selection.items() if key != "payload_sha256"
    }
    if _canonical_sha256(selection_payload) != selection_hash:
        _fail("dataset.selection payload hash mismatch")
    selected_ids = _unique_strings(
        selection.get("task_ids"), "dataset.selection.task_ids"
    )
    if len(selected_ids) != EXPECTED_TASK_COUNT:
        _fail(
            f"dataset selection must contain {EXPECTED_TASK_COUNT} tasks, "
            f"found {len(selected_ids)}"
        )

    raw_tasks = _list(dataset.get("tasks"), "dataset.tasks")
    indexed: dict[str, dict[str, Any]] = {}
    for index, raw_bundle in enumerate(raw_tasks):
        bundle = _object(raw_bundle, f"dataset.tasks[{index}]")
        task_id = _string(bundle.get("task_id"), f"dataset.tasks[{index}].task_id")
        if task_id in indexed:
            _fail(f"dataset contains duplicate task {task_id}")
        declared_hash = _sha256(
            bundle.get("bundle_sha256"), f"task {task_id}.bundle_sha256"
        )
        hash_input = {
            key: value for key, value in bundle.items() if key != "bundle_sha256"
        }
        if _canonical_sha256(hash_input) != declared_hash:
            _fail(f"task {task_id} bundle hash mismatch")
        indexed[task_id] = bundle
    if list(indexed) != selected_ids:
        _fail("dataset task order/set does not match the frozen selection")
    if schema_version == IMAGEGEN_SCHEMA_VERSION:
        _validate_imagegen_dataset(dataset, indexed)
    return indexed


def load_dataset(path: Path | None = None) -> tuple[Path, dict[str, Any]]:
    """Load and validate the top-level frozen runtime dataset."""
    resolved = resolve_dataset_path(path)
    dataset = _load_json(resolved)
    _validate_dataset(dataset)
    return resolved, dataset


def _validate_policy_record(value: Any, location: str) -> str:
    record = _object(value, location)
    text = _string(record.get("text"), f"{location}.text")
    digest = _sha256(record.get("sha256"), f"{location}.sha256")
    if _sha256_text(text) != digest:
        _fail(f"{location} hash mismatch")
    return text


def _validate_task_bundle(
    bundle: dict[str, Any],
    *,
    task: "Task",
    knowledge_base: "KnowledgeBase | None" = None,
) -> tuple[str, str, list[dict[str, Any]]]:
    task_id = _string(bundle.get("task_id"), "task.task_id")
    if task.id != task_id:
        _fail(f"requested task {task.id!r} does not match bundle {task_id!r}")
    required = _unique_strings(
        bundle.get("required_documents"), f"task {task_id}.required_documents"
    )
    if required != list(task.required_documents or []):
        _fail(f"task {task_id} required-document order drifted")

    policies = _object(bundle.get("policies"), f"task {task_id}.policies")
    full_policy = _validate_policy_record(
        policies.get("full_text"), f"task {task_id}.policies.full_text"
    )
    matched_policy = _validate_policy_record(
        policies.get("matched_ft"), f"task {task_id}.policies.matched_ft"
    )
    matched_record = _object(
        policies.get("matched_ft"), f"task {task_id}.policies.matched_ft"
    )
    if matched_record.get("shared_by_arms") != [
        "partitioned_text",
        "complementary",
    ]:
        _fail(
            f"task {task_id}.policies.matched_ft.shared_by_arms must bind "
            "partitioned_text and complementary in that order"
        )

    raw_documents = _list(bundle.get("documents"), f"task {task_id}.documents")
    if len(raw_documents) != len(required):
        _fail(f"task {task_id} document count does not match required_documents")
    documents: list[dict[str, Any]] = []
    for ordinal, (raw_document, expected_id) in enumerate(
        zip(raw_documents, required, strict=True)
    ):
        location = f"task {task_id}.documents[{ordinal}]"
        document = _object(raw_document, location)
        if document.get("ordinal") != ordinal:
            _fail(f"{location}.ordinal must be {ordinal}")
        document_id = _string(document.get("document_id"), f"{location}.document_id")
        if document_id != expected_id:
            _fail(f"{location}.document_id must be {expected_id!r}")
        binding_hash = _sha256(
            document.get("binding_sha256"), f"{location}.binding_sha256"
        )
        binding_payload = {
            key: value
            for key, value in document.items()
            if key not in {"ordinal", "binding_sha256"}
        }
        if _canonical_sha256(binding_payload) != binding_hash:
            _fail(f"{location} binding hash mismatch")

        source = _object(document.get("source"), f"{location}.source")
        title = _string(source.get("title"), f"{location}.source.title")
        title_hash = _sha256(
            source.get("title_sha256"), f"{location}.source.title_sha256"
        )
        content_hash = _sha256(
            source.get("content_sha256"), f"{location}.source.content_sha256"
        )
        _sha256(source.get("file_sha256"), f"{location}.source.file_sha256")
        _string(source.get("path"), f"{location}.source.path")
        if _sha256_text(title) != title_hash:
            _fail(f"{location} source title hash mismatch")

        if knowledge_base is not None:
            current = knowledge_base.documents.get(document_id)
            if current is None:
                _fail(f"{location} is absent from the current knowledge base")
            if current.title != title or _sha256_text(current.title) != title_hash:
                _fail(f"{location} source title drifted")
            if _sha256_text(current.content) != content_hash:
                _fail(f"{location} source content drifted")

        atoms = _object(document.get("atom_ids"), f"{location}.atom_ids")
        all_atoms = _unique_strings(atoms.get("all"), f"{location}.atom_ids.all")
        text_atoms = _unique_strings(atoms.get("text"), f"{location}.atom_ids.text")
        image_atoms = _unique_strings(atoms.get("image"), f"{location}.atom_ids.image")
        if set(text_atoms) & set(image_atoms):
            _fail(f"{location} F_T/F_I intersection is not empty")
        if set(text_atoms) | set(image_atoms) != set(all_atoms):
            _fail(f"{location} F_T/F_I union does not equal F")

        ft_projection = _object(
            document.get("ft_text_projection"), f"{location}.ft_text_projection"
        )
        ft_text = _string(
            ft_projection.get("text"), f"{location}.ft_text_projection.text"
        )
        if _sha256_text(ft_text) != _sha256(
            ft_projection.get("sha256"), f"{location}.ft_text_projection.sha256"
        ):
            _fail(f"{location} F_T text projection hash mismatch")
        ft_atoms = _unique_strings(
            ft_projection.get("atom_ids"), f"{location}.ft_text_projection.atom_ids"
        )
        if ft_atoms != text_atoms:
            _fail(f"{location} matched policy F_T atoms differ from text atoms")

        bridge = _object(document.get("bridge"), f"{location}.bridge")
        bridge_text = _string(bridge.get("text"), f"{location}.bridge.text")
        if _sha256_text(bridge_text) != _sha256(
            bridge.get("sha256"), f"{location}.bridge.sha256"
        ):
            _fail(f"{location} bridge hash mismatch")

        fi_projection = _object(
            document.get("fi_text_projection"), f"{location}.fi_text_projection"
        )
        fi_text = _string(
            fi_projection.get("text"), f"{location}.fi_text_projection.text"
        )
        if _sha256_text(fi_text) != _sha256(
            fi_projection.get("sha256"), f"{location}.fi_text_projection.sha256"
        ):
            _fail(f"{location} F_I text projection hash mismatch")
        fi_atoms = _unique_strings(
            fi_projection.get("atom_ids"), f"{location}.fi_text_projection.atom_ids"
        )
        if fi_atoms != image_atoms:
            _fail(f"{location} partitioned-text F_I atoms differ from image atoms")
        visible_occurrences = _list(
            fi_projection.get("visible_source_occurrences"),
            f"{location}.fi_text_projection.visible_source_occurrences",
        )
        if not visible_occurrences:
            _fail(f"{location} F_I visible occurrences must not be empty")
        visible_hash = _sha256(
            fi_projection.get("visible_source_occurrences_sha256"),
            f"{location}.fi_text_projection.visible_source_occurrences_sha256",
        )
        if _canonical_sha256(visible_occurrences) != visible_hash:
            _fail(f"{location} F_I visible-occurrence hash mismatch")
        visible_texts: list[str] = []
        visible_coverage: set[str] = set()
        for occurrence_index, raw_occurrence in enumerate(visible_occurrences):
            occurrence_location = (
                f"{location}.fi_text_projection.visible_source_occurrences"
                f"[{occurrence_index}]"
            )
            occurrence = _object(raw_occurrence, occurrence_location)
            source_occurrence = _object(
                occurrence.get("source_occurrence"),
                f"{occurrence_location}.source_occurrence",
            )
            visible_texts.append(
                _string(
                    source_occurrence.get("exact_text"),
                    f"{occurrence_location}.source_occurrence.exact_text",
                )
            )
            covered_atoms = _unique_strings(
                occurrence.get("covered_atom_ids"),
                f"{occurrence_location}.covered_atom_ids",
            )
            if not set(covered_atoms) <= set(image_atoms):
                _fail(f"{occurrence_location} covers atoms outside F_I")
            visible_coverage.update(covered_atoms)
        if visible_coverage != set(image_atoms):
            _fail(f"{location} visible strings do not cover every F_I atom")
        if fi_text != "\n".join(visible_texts):
            _fail(
                f"{location} partitioned-text payload differs from bitmap-visible "
                "string order/repetition"
            )

        coverage: list[str] = []
        assets = _list(document.get("assets"), f"{location}.assets")
        if not assets:
            _fail(f"{location}.assets must not be empty")
        asset_ids: list[str] = []
        for asset_index, raw_asset in enumerate(assets):
            asset_location = f"{location}.assets[{asset_index}]"
            asset = _object(raw_asset, asset_location)
            asset_ids.append(
                _string(asset.get("asset_id"), f"{asset_location}.asset_id")
            )
            if asset.get("mime_type") != "image/png":
                _fail(f"{asset_location}.mime_type must be 'image/png'")
            _string(asset.get("path"), f"{asset_location}.path")
            _sha256(asset.get("sha256"), f"{asset_location}.sha256")
            _positive_int(asset.get("width"), f"{asset_location}.width")
            _positive_int(asset.get("height"), f"{asset_location}.height")
            if (
                _sha256(
                    asset.get("visible_source_occurrences_sha256"),
                    f"{asset_location}.visible_source_occurrences_sha256",
                )
                != visible_hash
            ):
                _fail(
                    f"{asset_location} bitmap-visible occurrence binding differs "
                    "from partitioned text"
                )
            covered = _unique_strings(
                asset.get("covered_atom_ids"),
                f"{asset_location}.covered_atom_ids",
            )
            if not set(covered) <= set(image_atoms):
                _fail(f"{asset_location} covers atoms outside F_I")
            coverage.extend(covered)
        if len(asset_ids) != len(set(asset_ids)):
            _fail(f"{location}.assets contains duplicate asset IDs")
        if len(coverage) != len(set(coverage)) or set(coverage) != set(image_atoms):
            _fail(f"{location} bitmap coverage must cover every F_I atom exactly once")
        documents.append(document)
    return full_policy, matched_policy, documents


def policy_for_task(
    knowledge_base: "KnowledgeBase",
    task: "Task | None",
    *,
    arm: str | None = None,
    dataset_path: Path | None = None,
) -> str:
    """Return the frozen arm-specific policy used by a retrieval variant."""
    if task is None:
        _fail("selected complementary retrieval requires the current task")
    selected_arm = resolve_arm(arm)
    _, dataset = load_dataset(dataset_path)
    bundles = _validate_dataset(dataset)
    if task.id not in bundles:
        _fail(f"task {task.id!r} is outside the frozen selected cohort")
    full_policy, matched_policy, _ = _validate_task_bundle(
        bundles[task.id], task=task, knowledge_base=knowledge_base
    )
    return full_policy if selected_arm == "full_text" else matched_policy


def _resolve_asset(asset_root: Path, raw_path: str, location: str) -> Path:
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        _fail(f"{location} must be a contained relative path")
    try:
        resolved = (asset_root / relative).resolve(strict=True)
    except OSError as exc:
        _fail(f"cannot resolve {location}: {exc}")
    try:
        resolved.relative_to(asset_root)
    except ValueError:
        _fail(f"{location} escapes asset root {asset_root}")
    if not resolved.is_file():
        _fail(f"{location} is not a regular file")
    return resolved


def _validate_imagegen_external_bindings(
    dataset: dict[str, Any], *, asset_root: Path
) -> None:
    """Resolve the runtime contract against repository evidence and bytes."""
    contract = _object(dataset.get("imagegen_contract"), "dataset.imagegen_contract")

    def bound_json(value: Any, location: str) -> tuple[Path, dict[str, Any]]:
        record = _object(value, location)
        path = _resolve_asset(
            asset_root,
            _string(record.get("path"), f"{location}.path"),
            f"{location}.path",
        )
        try:
            payload = path.read_bytes()
            parsed = json.loads(payload.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            _fail(f"cannot read bound JSON {location}: {exc}")
        if _sha256_bytes(payload) != _sha256(
            record.get("sha256"), f"{location}.sha256"
        ):
            _fail(f"{location} file hash mismatch")
        parsed = _object(parsed, f"{location} payload")
        if parsed.get("schema_version") != record.get("schema_version"):
            _fail(f"{location} file schema mismatch")
        return path, parsed

    _, annotation = bound_json(
        contract.get("annotation_bundle_binding"),
        "dataset.imagegen_contract.annotation_bundle_binding",
    )
    _, manifest = bound_json(
        contract.get("asset_manifest_binding"),
        "dataset.imagegen_contract.asset_manifest_binding",
    )
    _, final_review = bound_json(
        contract.get("final_review_binding"),
        "dataset.imagegen_contract.final_review_binding",
    )
    if final_review.get("review_id") != contract["final_review_binding"].get(
        "review_id"
    ):
        _fail("external final review ID differs from imagegen_contract")
    if _canonical_sha256(
        _list(manifest.get("entries"), "external manifest.entries")
    ) != contract.get("manifest_entries_sha256"):
        _fail("external manifest entry hash differs from imagegen_contract")
    if _canonical_sha256(
        _list(final_review.get("documents"), "external final review.documents")
    ) != contract.get("review_rows_sha256"):
        _fail("external final review row hash differs from imagegen_contract")
    manifest_annotation = _object(
        manifest.get("annotation_bundle"), "external manifest.annotation_bundle"
    )
    annotation_binding = contract["annotation_bundle_binding"]
    for field in ("path", "sha256", "schema_version"):
        if manifest_annotation.get(field) != annotation_binding.get(field):
            _fail(f"external manifest annotation {field} binding mismatch")
    review_bindings = _object(
        final_review.get("bindings"), "external final review.bindings"
    )
    for name, expected in (
        ("annotation_bundle", annotation_binding),
        ("asset_manifest", contract["asset_manifest_binding"]),
    ):
        observed = _object(
            review_bindings.get(name), f"external final review.bindings.{name}"
        )
        for field in ("path", "sha256", "schema_version"):
            if observed.get(field) != expected.get(field):
                _fail(f"external final review {name}.{field} binding mismatch")

    shard_payloads: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(contract["shard_manifest_bindings"]):
        record = _object(raw, f"imagegen shard binding {index}")
        _, payload = bound_json(record, f"imagegen shard binding {index}")
        shard_payloads[record["path"]] = payload
    reviewed_ids: list[str] = []
    for index, raw in enumerate(contract["independent_review_bindings"]):
        record = _object(raw, f"imagegen review binding {index}")
        _, payload = bound_json(record, f"imagegen review binding {index}")
        reviewer = _object(payload.get("reviewer"), f"imagegen review {index}.reviewer")
        if (
            reviewer.get("id") != record.get("reviewer_id")
            or reviewer.get("reviewed_shard") != record.get("reviewed_shard")
            or reviewer.get("reviewer_independence") is not True
            or reviewer.get("did_not_generate_reviewed_assets") is not True
        ):
            _fail(f"external independent review {index} identity mismatch")
        bound_shard = _object(
            record.get("bound_shard_manifest"),
            f"imagegen review binding {index}.bound_shard_manifest",
        )
        shard = shard_payloads.get(bound_shard.get("path"))
        if shard is None:
            _fail(f"external independent review {index} binds no frozen shard")
        shard_identity = _object(shard.get("shard"), f"external shard {index}.shard")
        if shard_identity.get("id") != record.get("reviewed_shard"):
            _fail(f"external independent review {index} shard identity mismatch")
        review_documents = _list(
            payload.get("documents"), f"external independent review {index}.documents"
        )
        shard_documents = _list(
            shard.get("documents"), f"external shard {index}.documents"
        )
        review_ids = [row.get("document_id") for row in review_documents]
        if len(review_ids) != 20 or review_ids != [
            row.get("document_id") for row in shard_documents
        ]:
            _fail(f"external independent review {index} shard closure mismatch")
        reviewed_ids.extend(review_ids)
    if reviewed_ids != contract["unique_document_ids"]:
        _fail("external independent review union/order differs from 60 documents")

    annotation_rows = _list(
        annotation.get("document_index"), "external annotation.document_index"
    )
    manifest_rows = _list(manifest.get("entries"), "external manifest.entries")
    review_rows = _list(
        final_review.get("documents"), "external final review.documents"
    )
    unique_ids = contract["unique_document_ids"]
    if any(
        len(rows) != EXPECTED_UNIQUE_DOCUMENT_COUNT
        for rows in (
            annotation_rows,
            manifest_rows,
            review_rows,
        )
    ):
        _fail("external annotation/manifest/review must each contain 60 rows")
    if any(
        [row.get("document_id") for row in rows] != unique_ids
        for rows in (
            annotation_rows,
            manifest_rows,
            review_rows,
        )
    ):
        _fail("external annotation/manifest/review document order differs")

    unique_runtime_documents: dict[str, dict[str, Any]] = {}
    for bundle in dataset["tasks"]:
        for document in bundle["documents"]:
            unique_runtime_documents.setdefault(document["document_id"], document)
    for document_id, annotation_row, manifest_row, review_row in zip(
        unique_ids,
        annotation_rows,
        manifest_rows,
        review_rows,
        strict=True,
    ):
        runtime_document = unique_runtime_documents[document_id]
        runtime_asset = _object(
            runtime_document["assets"][0], f"runtime {document_id}.asset"
        )
        rebind = _object(
            runtime_asset.get("realization_rebind"), f"runtime {document_id}.rebind"
        )
        annotation_path = _resolve_asset(
            asset_root,
            _string(annotation_row.get("path"), f"annotation {document_id}.path"),
            f"annotation {document_id}.path",
        )
        annotation_bytes = annotation_path.read_bytes()
        if _sha256_bytes(annotation_bytes) != annotation_row.get("sha256"):
            _fail(f"annotation sidecar {document_id} hash mismatch")
        annotation_document = _object(
            json.loads(annotation_bytes.decode("utf-8")),
            f"annotation sidecar {document_id}",
        )
        projection = _object(
            annotation_document.get("image_projection"),
            f"annotation {document_id}.image_projection",
        )
        if (
            projection.get("asset_id") != runtime_asset.get("asset_id")
            or projection.get("planned_path")
            != runtime_asset.get("annotation_planned_path")
            or projection.get("binding_sha256")
            != rebind.get("annotation_image_projection_binding_sha256")
            or projection.get("visible_source_occurrences_sha256")
            != rebind.get("visible_source_occurrences_sha256")
        ):
            _fail(
                f"annotation-to-imagegen realization rebind mismatch for {document_id}"
            )
        if _canonical_sha256(manifest_row) != rebind.get("manifest_entry_sha256"):
            _fail(f"manifest entry hash mismatch for runtime document {document_id}")
        if _canonical_sha256(review_row) != rebind.get("review_row_sha256"):
            _fail(f"review row hash mismatch for runtime document {document_id}")
        manifest_asset = _object(
            manifest_row.get("asset"), f"external manifest {document_id}.asset"
        )
        review_asset = _object(
            review_row.get("asset"), f"external review {document_id}.asset"
        )
        if (
            manifest_row.get("asset_id") != runtime_asset.get("asset_id")
            or any(
                manifest_asset.get(field) != runtime_asset.get(field)
                for field in ("path", "sha256", "mime_type", "width", "height")
            )
            or review_asset.get("asset_id") != runtime_asset.get("asset_id")
            or review_asset.get("path") != runtime_asset.get("path")
            or review_asset.get("sha256") != runtime_asset.get("sha256")
        ):
            _fail(f"external asset binding mismatch for runtime document {document_id}")


def _png_dimensions(payload: bytes, location: str) -> tuple[int, int]:
    if len(payload) < 24 or payload[:8] != _PNG_SIGNATURE:
        _fail(f"{location} is not a PNG")
    if payload[12:16] != b"IHDR" or struct.unpack(">I", payload[8:12])[0] != 13:
        _fail(f"{location} has no canonical first IHDR chunk")
    width, height = struct.unpack(">II", payload[16:24])
    if width <= 0 or height <= 0:
        _fail(f"{location} has invalid dimensions {width}x{height}")
    return width, height


def load_task_runtime(
    *,
    task: "Task",
    domain_policy: str,
    arm: str | None = None,
    dataset_path: Path | None = None,
    asset_root: Path | None = None,
) -> RuntimeTask:
    """Validate and materialize one task for the selected matched agent."""
    selected_arm = resolve_arm(arm)
    resolved_dataset, dataset = load_dataset(dataset_path)
    bundles = _validate_dataset(dataset)
    if task.id not in bundles:
        _fail(f"task {task.id!r} is outside the frozen selected cohort")
    full_policy, matched_policy, documents = _validate_task_bundle(
        bundles[task.id], task=task
    )
    expected_policy = full_policy if selected_arm == "full_text" else matched_policy
    if domain_policy != expected_policy:
        _fail(f"task {task.id} environment policy does not match arm {selected_arm!r}")

    resolved_root = resolve_asset_root(resolved_dataset, asset_root)
    if dataset.get("schema_version") == IMAGEGEN_SCHEMA_VERSION:
        _validate_imagegen_external_bindings(dataset, asset_root=resolved_root)
    runtime_documents: list[RuntimeDocument] = []
    for ordinal, document in enumerate(documents):
        image_payloads: list[bytes] = []
        for asset_index, asset in enumerate(document["assets"]):
            location = f"task {task.id}.documents[{ordinal}].assets[{asset_index}]"
            resolved = _resolve_asset(
                resolved_root, _string(asset.get("path"), f"{location}.path"), location
            )
            try:
                payload = resolved.read_bytes()
            except OSError as exc:
                _fail(f"cannot read {location}: {exc}")
            expected_hash = _sha256(asset.get("sha256"), f"{location}.sha256")
            if _sha256_bytes(payload) != expected_hash:
                _fail(f"{location} bitmap hash mismatch")
            expected_dimensions = (
                _positive_int(asset.get("width"), f"{location}.width"),
                _positive_int(asset.get("height"), f"{location}.height"),
            )
            if _png_dimensions(payload, location) != expected_dimensions:
                _fail(f"{location} bitmap dimensions drifted")
            image_payloads.append(payload)
        runtime_documents.append(
            RuntimeDocument(
                document_id=document["document_id"],
                bridge_text=document["bridge"]["text"],
                fi_text=document["fi_text_projection"]["text"],
                image_payloads=tuple(image_payloads),
            )
        )
    return RuntimeTask(
        arm=selected_arm,
        task_id=task.id,
        policy=expected_policy,
        documents=tuple(runtime_documents),
        dataset_path=resolved_dataset,
        asset_root=resolved_root,
    )
