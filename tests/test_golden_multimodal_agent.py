import base64
import hashlib
import json

import pytest

from tau2.agent.golden_multimodal_agent import (
    EXPECTED_SCHEMA_VERSION,
    EXPECTED_SETTING,
    GOLDEN_MULTIMODAL_ENVELOPE,
    GoldenMultimodalAgent,
    GoldenMultimodalDatasetError,
)
from tau2.agent.llm_agent import LLMAgent
from tau2.data_model.message import UserMessage
from tau2.data_model.tasks import Task
from tau2.registry import registry
from tau2.utils.llm_utils import _format_messages_for_logging, to_litellm_messages

PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUB"
    "AScY42YAAAAASUVORK5CYII="
)


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


def _make_document(index: int, document_id: str, path: str, payload: bytes) -> dict:
    title = f"Document {index}"
    anchors = [f"source anchor {index}"]
    return {
        "ordinal": index,
        "document_id": document_id,
        "title": title,
        "title_sha256": _sha256_text(title),
        "content_sha256": _sha256_text(f"full content {index}"),
        "text_mode": EXPECTED_SETTING["text_mode"],
        "image_mode": EXPECTED_SETTING["image_mode"],
        "source_anchors": anchors,
        "source_anchors_sha256": _canonical_sha256(anchors),
        "asset_path": path,
        "asset_sha256": _sha256_bytes(payload),
        "mime_type": "image/png",
        "width": 1,
        "height": 1,
        "semantic_verdict": "PASS" if index == 0 else "WARN",
        "audit_provenance": {"review": "fixture"},
    }


def _write_dataset(tmp_path, *, policy="full golden policy"):
    assets = tmp_path / "assets"
    assets.mkdir(parents=True)
    payloads = [PNG_1X1 + b"first", PNG_1X1 + b"second"]
    document_ids = ["doc_a", "doc_b"]
    documents = []
    for index, (document_id, payload) in enumerate(zip(document_ids, payloads)):
        relative_path = f"assets/{document_id}.png"
        (tmp_path / relative_path).write_bytes(payload)
        documents.append(_make_document(index, document_id, relative_path, payload))

    bundle = {
        "task_id": "task_001",
        "required_documents": document_ids,
        "documents": documents,
        "policy_sha256": _sha256_text(policy),
        "required_documents_block_sha256": _sha256_text("required docs block"),
    }
    bundle["bundle_sha256"] = _canonical_sha256(bundle)
    dataset = {
        "schema_version": EXPECTED_SCHEMA_VERSION,
        "setting": dict(EXPECTED_SETTING),
        "provenance": {"fixture": True},
        "tasks": [bundle],
    }
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
    task = Task.model_construct(id="task_001", required_documents=list(document_ids))
    return dataset_path, dataset, task, payloads


def _rehash_bundle(dataset: dict) -> None:
    bundle = dataset["tasks"][0]
    bundle["bundle_sha256"] = _canonical_sha256(
        {key: value for key, value in bundle.items() if key != "bundle_sha256"}
    )


def test_user_message_multiple_images_preserve_order_and_exclude_serialization():
    message = UserMessage(
        role="user",
        content="ordered images",
        image_content="legacy-is-ignored",
        image_pages=["first-page", "second-page"],
    )

    converted = to_litellm_messages([message])

    assert converted == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "ordered images"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,first-page"},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,second-page"},
                },
            ],
        }
    ]
    assert "image_pages" not in message.model_dump()


def test_user_message_single_image_remains_backward_compatible():
    message = UserMessage(
        role="user", content="legacy image", image_content="only-page"
    )

    converted = to_litellm_messages([message])

    assert converted[0]["content"][1]["image_url"]["url"] == (
        "data:image/png;base64,only-page"
    )


def test_logging_redacts_nested_image_data_without_mutating_request():
    request_messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "visible"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,secret-image-payload"},
                },
            ],
        }
    ]

    formatted = _format_messages_for_logging(request_messages)

    assert formatted[0]["content"][1]["image_url"]["url"] == (
        "data:image/png;base64,<redacted>"
    )
    assert request_messages[0]["content"][1]["image_url"]["url"] == (
        "data:image/png;base64,secret-image-payload"
    )


def test_agent_validates_bundle_and_adds_one_private_ordered_envelope(tmp_path):
    policy = "full golden policy"
    dataset_path, _, task, payloads = _write_dataset(tmp_path, policy=policy)
    history = [UserMessage(role="user", content="existing history")]
    agent = GoldenMultimodalAgent(
        tools=[],
        domain_policy=policy,
        task=task,
        llm="test-model",
        dataset_path=dataset_path,
        asset_root=tmp_path,
    )

    state = agent.get_init_state(history)

    baseline = LLMAgent(
        tools=[], domain_policy=policy, llm="test-model"
    ).get_init_state(history)
    assert state.system_messages == baseline.system_messages
    assert len(state.messages) == len(history) + 1
    envelope = state.messages[0]
    assert isinstance(envelope, UserMessage)
    assert envelope.content == GOLDEN_MULTIMODAL_ENVELOPE
    assert envelope.image_pages == [
        base64.b64encode(payload).decode("ascii") for payload in payloads
    ]
    assert state.messages[1:] == history
    assert len(history) == 1
    assert history[0].content == "existing history"
    assert "image_pages" not in envelope.model_dump()
    assert registry.get_agent_factory("llm_agent_golden_multimodal") is not None


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (
            lambda dataset: dataset["tasks"][0]["required_documents"].reverse(),
            "required-document order",
        ),
        (
            lambda dataset: dataset["tasks"][0]["documents"][0].update(
                {"asset_path": "../outside.png"}
            ),
            "contained relative path",
        ),
        (
            lambda dataset: dataset["tasks"][0]["documents"][0].update({"width": 2}),
            "image dimensions",
        ),
    ],
)
def test_agent_rejects_order_path_and_dimension_tampering(tmp_path, mutation, error):
    dataset_path, dataset, task, _ = _write_dataset(tmp_path)
    mutation(dataset)
    _rehash_bundle(dataset)
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")

    with pytest.raises(GoldenMultimodalDatasetError, match=error):
        GoldenMultimodalAgent(
            tools=[],
            domain_policy="full golden policy",
            task=task,
            llm="test-model",
            dataset_path=dataset_path,
            asset_root=tmp_path,
        )


def test_agent_rejects_bundle_asset_and_policy_hash_mismatches(tmp_path):
    dataset_path, dataset, task, _ = _write_dataset(tmp_path)
    dataset["tasks"][0]["documents"][0]["semantic_verdict"] = "FAIL"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
    with pytest.raises(GoldenMultimodalDatasetError, match="bundle hash mismatch"):
        GoldenMultimodalAgent(
            tools=[],
            domain_policy="full golden policy",
            task=task,
            llm="test-model",
            dataset_path=dataset_path,
            asset_root=tmp_path,
        )

    dataset_path, dataset, task, _ = _write_dataset(tmp_path / "asset-case")
    asset_path = tmp_path / "asset-case" / "assets" / "doc_a.png"
    asset_path.write_bytes(asset_path.read_bytes() + b"tampered")
    with pytest.raises(GoldenMultimodalDatasetError, match="asset hash mismatch"):
        GoldenMultimodalAgent(
            tools=[],
            domain_policy="full golden policy",
            task=task,
            llm="test-model",
            dataset_path=dataset_path,
            asset_root=tmp_path / "asset-case",
        )

    dataset_path, _, task, _ = _write_dataset(tmp_path / "policy-case")
    with pytest.raises(GoldenMultimodalDatasetError, match="policy hash mismatch"):
        GoldenMultimodalAgent(
            tools=[],
            domain_policy="changed golden policy",
            task=task,
            llm="test-model",
            dataset_path=dataset_path,
            asset_root=tmp_path / "policy-case",
        )
