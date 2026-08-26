"""Regression tests for the frozen selected-cohort matched three-arm agent."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tau2.agent.llm_agent import LLMAgent
from tau2.agent.selected_complementary_agent import (
    MATCHED_SUPPLEMENT_ENVELOPE,
    SelectedComplementaryAgent,
)
from tau2.data_model.message import UserMessage
from tau2.data_model.simulation import TextRunConfig
from tau2.data_model.tasks import Task
from tau2.domains.banking_knowledge.selected_complementary import (
    ARMS,
    EXPECTED_TASK_COUNT,
    IMAGEGEN_SCHEMA_VERSION,
    IMAGEGEN_SETTING_ID,
    PRIMARY_ARMS,
    SCHEMA_VERSION,
    SETTING_ID,
    SelectedComplementaryRuntimeError,
    policy_for_task,
)
from tau2.registry import registry
from tau2.runner import helpers as runner_helpers

PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUB"
    "AScY42YAAAAASUVORK5CYII="
)
SELECTED_IDS = [f"task_{index:03d}" for index in range(1, 20)]
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _canonical_sha256(value) -> str:
    return _sha256_text(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


def _policy(text: str, *, matched: bool = False) -> dict:
    record = {"text": text, "sha256": _sha256_text(text)}
    if matched:
        record["shared_by_arms"] = ["partitioned_text", "complementary"]
    return record


def _document(document_id: str, asset_path: str, payload: bytes) -> dict:
    title = "Frozen Fixture Document"
    bridge = f"Document ID: {document_id}"
    ft_text = "Text-only atom"
    fi_text = "Image-only atom"
    visible_occurrences = [
        {
            "source_occurrence": {
                "field": "content",
                "line_start": 2,
                "line_end": 2,
                "char_start": 15,
                "char_end": 30,
                "exact_text": fi_text,
            },
            "covered_atom_ids": [f"{document_id}.a02"],
        }
    ]
    visible_hash = _canonical_sha256(visible_occurrences)
    document = {
        "document_id": document_id,
        "source": {
            "path": f"documents/{document_id}.json",
            "file_sha256": "0" * 64,
            "title": title,
            "title_sha256": _sha256_text(title),
            "content_sha256": _sha256_text(f"{ft_text}\n{fi_text}"),
        },
        "bridge": {"text": bridge, "sha256": _sha256_text(bridge)},
        "atom_ids": {
            "all": [f"{document_id}.a01", f"{document_id}.a02"],
            "text": [f"{document_id}.a01"],
            "image": [f"{document_id}.a02"],
        },
        "ft_text_projection": {
            "renderer": "fixture-v1",
            "text": ft_text,
            "sha256": _sha256_text(ft_text),
            "atom_ids": [f"{document_id}.a01"],
        },
        "fi_text_projection": {
            "renderer": "fixture-v1",
            "text": fi_text,
            "sha256": _sha256_text(fi_text),
            "atom_ids": [f"{document_id}.a02"],
            "visible_source_occurrences": visible_occurrences,
            "visible_source_occurrences_sha256": visible_hash,
        },
        "assets": [
            {
                "asset_id": f"{document_id}.image01",
                "path": asset_path,
                "sha256": _sha256_bytes(payload),
                "mime_type": "image/png",
                "width": 1,
                "height": 1,
                "covered_atom_ids": [f"{document_id}.a02"],
                "visible_source_occurrences_sha256": visible_hash,
            }
        ],
    }
    document["binding_sha256"] = _canonical_sha256(document)
    document["ordinal"] = 0
    return document


def _rehash_document(document: dict) -> None:
    document["binding_sha256"] = _canonical_sha256(
        {
            key: value
            for key, value in document.items()
            if key not in {"ordinal", "binding_sha256"}
        }
    )


def _rehash_bundle(bundle: dict) -> None:
    bundle["bundle_sha256"] = _canonical_sha256(
        {key: value for key, value in bundle.items() if key != "bundle_sha256"}
    )


def _write_dataset(tmp_path):
    payload = PNG_1X1 + b"selected-complementary-fixture"
    asset_path = "assets/doc_fixture.png"
    resolved_asset = tmp_path / asset_path
    resolved_asset.parent.mkdir(parents=True)
    resolved_asset.write_bytes(payload)

    bundles = []
    for task_id in SELECTED_IDS:
        document_id = f"doc_{task_id}"
        bundle = {
            "task_id": task_id,
            "required_documents": [document_id],
            "policies": {
                "full_text": _policy(f"full policy for {task_id}"),
                "matched_ft": _policy(
                    f"matched F_T policy for {task_id}", matched=True
                ),
            },
            "documents": [_document(document_id, asset_path, payload)],
        }
        _rehash_bundle(bundle)
        bundles.append(bundle)

    selection = {
        "schema_version": "fixture-selection-v1",
        "selection_id": "fixture-selection",
        "task_ids": SELECTED_IDS,
    }
    selection["payload_sha256"] = _canonical_sha256(selection)
    dataset = {
        "schema_version": SCHEMA_VERSION,
        "status": "frozen_run_eligible",
        "setting": {
            "id": SETTING_ID,
            "task_count": EXPECTED_TASK_COUNT,
            "arms": list(ARMS),
            "primary_contrast": list(PRIMARY_ARMS),
            "allocation": {
                "text_atoms": "F_T",
                "image_atoms": "F_I",
                "intersection": "empty",
                "union": "F",
            },
        },
        "selection": selection,
        "provenance": {"fixture": True},
        "tasks": bundles,
    }
    dataset_path = tmp_path / "runtime.json"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
    task = Task.model_construct(
        id=SELECTED_IDS[0],
        required_documents=[f"doc_{SELECTED_IDS[0]}"],
    )
    return dataset_path, dataset, task, payload


def _agent(tmp_path, task, dataset_path, arm: str) -> SelectedComplementaryAgent:
    policy = (
        f"full policy for {task.id}"
        if arm == "full_text"
        else f"matched F_T policy for {task.id}"
    )
    return SelectedComplementaryAgent(
        tools=[],
        domain_policy=policy,
        task=task,
        llm="test-model",
        dataset_path=dataset_path,
        asset_root=tmp_path,
        arm=arm,
    )


def test_three_arms_preserve_the_matched_contract(tmp_path) -> None:
    dataset_path, _, task, payload = _write_dataset(tmp_path)
    history = [UserMessage(role="user", content="caller-owned history")]

    full = _agent(tmp_path, task, dataset_path, "full_text").get_init_state(history)
    partitioned = _agent(
        tmp_path, task, dataset_path, "partitioned_text"
    ).get_init_state(history)
    complementary = _agent(
        tmp_path, task, dataset_path, "complementary"
    ).get_init_state(history)

    full_baseline = LLMAgent(
        tools=[], domain_policy=f"full policy for {task.id}", llm="test-model"
    ).get_init_state(history)
    assert full == full_baseline
    assert partitioned.system_messages == complementary.system_messages
    assert partitioned.system_messages != full.system_messages

    partitioned_envelope = partitioned.messages[0]
    complementary_envelope = complementary.messages[0]
    assert isinstance(partitioned_envelope, UserMessage)
    assert isinstance(complementary_envelope, UserMessage)
    assert partitioned_envelope == complementary_envelope
    assert partitioned_envelope.content == MATCHED_SUPPLEMENT_ENVELOPE
    assert partitioned_envelope.image_pages is None

    partitioned_document = partitioned.messages[1]
    complementary_document = complementary.messages[1]
    assert isinstance(partitioned_document, UserMessage)
    assert isinstance(complementary_document, UserMessage)
    assert partitioned_document.content == (
        f"Document ID: doc_{task.id}\n\nImage-only atom"
    )
    assert partitioned_document.image_pages is None
    assert complementary_document.content == (f"Document ID: doc_{task.id}")
    assert complementary_document.image_pages == [
        base64.b64encode(payload).decode("ascii")
    ]
    assert partitioned.messages[2:] == complementary.messages[2:] == history
    assert history == [UserMessage(role="user", content="caller-owned history")]
    assert registry.get_agent_factory("llm_agent_selected_complementary_v1")


def test_old_runtime_cannot_masquerade_as_imagegen_by_relabeling(
    tmp_path,
) -> None:
    dataset_path, dataset, task, _ = _write_dataset(tmp_path)
    dataset["schema_version"] = IMAGEGEN_SCHEMA_VERSION
    dataset["setting"]["id"] = IMAGEGEN_SETTING_ID
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")

    with pytest.raises(SelectedComplementaryRuntimeError, match="imagegen_contract"):
        _agent(tmp_path, task, dataset_path, "complementary")


def test_frozen_imagegen_runtime_passes_external_provenance_gate() -> None:
    dataset_path = (
        WORKSPACE_ROOT
        / "tauvision/data/banking_explorer/selected_complementary_runtime_imagegen_v1.json"
    )
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    task = Task.model_validate(
        json.loads(
            (
                WORKSPACE_ROOT
                / "tau2-bench/data/tau2/domains/banking_knowledge/tasks/task_001.json"
            ).read_text(encoding="utf-8")
        )
    )
    bundle = next(row for row in dataset["tasks"] if row["task_id"] == task.id)
    agent = SelectedComplementaryAgent(
        tools=[],
        domain_policy=bundle["policies"]["matched_ft"]["text"],
        task=task,
        llm="test-model",
        dataset_path=dataset_path,
        asset_root=WORKSPACE_ROOT,
        arm="complementary",
    )

    assert agent.runtime.arm == "complementary"
    assert agent.runtime.task_id == task.id
    assert len(agent.runtime.documents) == 4


def test_agent_fails_closed_on_policy_asset_and_cohort_drift(tmp_path) -> None:
    dataset_path, _, task, _ = _write_dataset(tmp_path)

    with pytest.raises(SelectedComplementaryRuntimeError, match="environment policy"):
        SelectedComplementaryAgent(
            tools=[],
            domain_policy="wrong policy",
            task=task,
            llm="test-model",
            dataset_path=dataset_path,
            asset_root=tmp_path,
            arm="complementary",
        )

    outside = Task.model_construct(id="task_999", required_documents=["doc_x"])
    with pytest.raises(SelectedComplementaryRuntimeError, match="outside"):
        _agent(tmp_path, outside, dataset_path, "full_text")

    (tmp_path / "assets/doc_fixture.png").write_bytes(PNG_1X1 + b"tampered")
    with pytest.raises(SelectedComplementaryRuntimeError, match="bitmap hash"):
        _agent(tmp_path, task, dataset_path, "complementary")


def test_agent_rejects_rehashed_ft_fi_overlap(tmp_path) -> None:
    dataset_path, dataset, task, _ = _write_dataset(tmp_path)
    document = dataset["tasks"][0]["documents"][0]
    document["atom_ids"]["text"].append(document["atom_ids"]["image"][0])
    document["ft_text_projection"]["atom_ids"].append(document["atom_ids"]["image"][0])
    _rehash_document(document)
    _rehash_bundle(dataset["tasks"][0])
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")

    with pytest.raises(SelectedComplementaryRuntimeError, match="intersection"):
        _agent(tmp_path, task, dataset_path, "partitioned_text")


def test_run_metadata_override_does_not_construct_a_taskless_environment(
    monkeypatch,
) -> None:
    placeholder = "(Policy is task-specific - see 'policy' field in each simulation)"
    config = TextRunConfig(
        domain="banking_knowledge",
        retrieval_config="golden_retrieval_selected_complementary_v1",
    )

    def _unexpected_environment_construction(*args, **kwargs):
        raise AssertionError("metadata must not construct a taskless environment")

    monkeypatch.setattr(
        runner_helpers,
        "get_environment_info",
        _unexpected_environment_construction,
    )

    info = runner_helpers.get_info(config, policy_override=placeholder)

    assert info.environment_info.domain_name == "banking_knowledge"
    assert info.environment_info.policy == placeholder
    assert info.environment_info.tool_defs is None


def test_task_policy_remains_required_and_hash_frozen(tmp_path) -> None:
    dataset_path, dataset, task, _ = _write_dataset(tmp_path)
    document_id = task.required_documents[0]
    source = dataset["tasks"][0]["documents"][0]["source"]
    knowledge_base = SimpleNamespace(
        documents={
            document_id: SimpleNamespace(
                title=source["title"],
                content="Text-only atom\nImage-only atom",
            )
        }
    )

    with pytest.raises(
        SelectedComplementaryRuntimeError,
        match="requires the current task",
    ):
        policy_for_task(
            knowledge_base,
            None,
            arm="partitioned_text",
            dataset_path=dataset_path,
        )

    assert (
        policy_for_task(
            knowledge_base,
            task,
            arm="partitioned_text",
            dataset_path=dataset_path,
        )
        == f"matched F_T policy for {task.id}"
    )

    knowledge_base.documents[document_id].content += "\ndrift"
    with pytest.raises(SelectedComplementaryRuntimeError, match="content drifted"):
        policy_for_task(
            knowledge_base,
            task,
            arm="partitioned_text",
            dataset_path=dataset_path,
        )
