"""Formal regression tests for the frozen two-arm in-place system experiment."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import litellm
import pytest

from tau2.agent.complementary_inplace_system_agent import (
    ComplementaryInplaceSystemAgent,
    build_inplace_system_content,
)
from tau2.agent.llm_agent import LLMAgent
from tau2.data_model.message import SystemMessage, UserMessage
from tau2.domains.banking_knowledge.complementary_inplace_system import (
    ARM_ENV,
    ARMS,
    EXPECTED_DOCUMENT_COUNT,
    EXPECTED_DOCUMENT_REFERENCE_COUNT,
    EXPECTED_TASK_COUNT,
    ComplementaryInplaceSystemError,
)
from tau2.domains.banking_knowledge.environment import (
    get_environment,
    get_knowledge_base,
    get_tasks,
)
from tau2.domains.banking_knowledge.retrieval import PROMPTS_DIR, golden_prompt
from tau2.registry import registry
from tau2.runner.build import build_agent
from tau2.utils.display import MarkdownDisplay
from tau2.utils.llm_utils import to_litellm_messages, validate_message

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = (
    WORKSPACE_ROOT
    / "tauvision/data/banking_explorer/complementary_inplace_system_v2"
    / "asset_manifest.json"
)
SELECTION_PATH = (
    WORKSPACE_ROOT
    / "tauvision/data/banking_explorer/complementary_5of5_selection_v1.json"
)
INLINE_PNG_PREFIX = "data:image/png;base64,"


@pytest.fixture(scope="module")
def frozen_inputs() -> SimpleNamespace:
    """Load the real frozen cohort once; no fixture images are copied."""
    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    selected_ids = selection["task_ids"]
    tasks_by_id = {task.id: task for task in get_tasks()}
    tasks = [tasks_by_id[task_id] for task_id in selected_ids]
    knowledge_base = get_knowledge_base()
    policies = {
        task.id: golden_prompt(PROMPTS_DIR / "required_docs.md", knowledge_base, task)
        for task in tasks
    }
    selection_rows = {row["task_id"]: row for row in selection["tasks"]}

    assert len(tasks) == EXPECTED_TASK_COUNT == 19
    assert list(selection_rows) == selected_ids
    assert (
        sum(len(row["required_documents"]) for row in selection["tasks"])
        == EXPECTED_DOCUMENT_REFERENCE_COUNT
        == 109
    )
    return SimpleNamespace(
        selection=selection,
        selection_rows=selection_rows,
        tasks=tasks,
        policies=policies,
    )


def _agent(frozen_inputs, task, arm: str) -> ComplementaryInplaceSystemAgent:
    return ComplementaryInplaceSystemAgent(
        tools=[],
        domain_policy=frozen_inputs.policies[task.id],
        task=task,
        llm="test-model",
        manifest_path=MANIFEST_PATH,
        asset_root=WORKSPACE_ROOT,
        arm=arm,
    )


def _decode_inline_png(part: dict) -> bytes:
    assert part["type"] == "image_url"
    url = part["image_url"]["url"]
    assert url.startswith(INLINE_PNG_PREFIX)
    return base64.b64decode(url[len(INLINE_PNG_PREFIX) :], validate=True)


def test_all_19_tasks_have_only_the_two_matched_arms_and_109_ordered_images(
    frozen_inputs,
) -> None:
    """Exercise the exact request projection for every task and both arms."""
    assert ARMS == ("inplace_text", "inplace_image")
    expected_reference_ids = [
        document_id
        for task_id in frozen_inputs.selection["task_ids"]
        for document_id in frozen_inputs.selection_rows[task_id]["required_documents"]
    ]
    observed_reference_ids: list[str] = []
    observed_payload_hashes: list[str] = []
    expected_payload_hashes: list[str] = []
    digest_by_document: dict[str, str] = {}

    for task in frozen_inputs.tasks:
        policy = frozen_inputs.policies[task.id]
        expected_documents = frozen_inputs.selection_rows[task.id]["required_documents"]
        assert task.required_documents == expected_documents

        history = [UserMessage(role="user", content=f"history for {task.id}")]
        history_before = [message.model_dump() for message in history]
        baseline = LLMAgent(
            tools=[], domain_policy=policy, llm="test-model"
        ).get_init_state(history)

        text_agent = _agent(frozen_inputs, task, "inplace_text")
        text_state = text_agent.get_init_state(history)
        assert text_agent.runtime.arm == "inplace_text"
        assert len(text_state.system_messages) == 1
        assert isinstance(text_state.system_messages[0].content, str)
        assert text_state.system_messages == baseline.system_messages
        text_content = text_state.system_messages[0].content
        baseline_content = baseline.system_messages[0].content
        assert isinstance(text_content, str)
        assert isinstance(baseline_content, str)
        assert text_content.encode("utf-8") == baseline_content.encode("utf-8")
        assert to_litellm_messages(
            text_state.system_messages + text_state.messages
        ) == to_litellm_messages(baseline.system_messages + baseline.messages)
        assert text_state.messages == history

        image_agent = _agent(frozen_inputs, task, "inplace_image")
        image_state = image_agent.get_init_state(history)
        assert image_agent.runtime.arm == "inplace_image"
        assert len(image_state.system_messages) == 1
        image_content = image_state.system_messages[0].content
        assert isinstance(image_content, list)
        assert image_state.messages == history
        assert image_state.messages is not history
        assert [message.model_dump() for message in history] == history_before
        assert [doc.document_id for doc in image_agent.runtime.documents] == (
            expected_documents
        )

        wire_messages = to_litellm_messages(
            image_state.system_messages + image_state.messages
        )
        assert wire_messages[0] == {"role": "system", "content": image_content}
        assert wire_messages[1:] == to_litellm_messages(history)
        assert sum(message["role"] == "user" for message in wire_messages) == len(
            history
        )

        reconstructed: list[str] = []
        task_payload_hashes: list[str] = []
        image_ordinal = 0
        for part in image_content:
            if part["type"] == "text":
                assert part["text"] != ""
                reconstructed.append(part["text"])
                continue

            document = image_agent.runtime.documents[image_ordinal]
            payload_digest = hashlib.sha256(_decode_inline_png(part)).hexdigest()
            assert payload_digest == document.image_sha256
            reconstructed.append(document.block)
            task_payload_hashes.append(payload_digest)
            observed_reference_ids.append(document.document_id)
            prior_digest = digest_by_document.setdefault(
                document.document_id, payload_digest
            )
            assert prior_digest == payload_digest
            image_ordinal += 1

        assert image_ordinal == len(expected_documents)
        assert "".join(reconstructed).encode("utf-8") == text_content.encode("utf-8")
        expected_task_hashes = [
            document.image_sha256 for document in image_agent.runtime.documents
        ]
        assert task_payload_hashes == expected_task_hashes
        observed_payload_hashes.extend(task_payload_hashes)
        expected_payload_hashes.extend(expected_task_hashes)

    assert observed_reference_ids == expected_reference_ids
    assert len(observed_reference_ids) == EXPECTED_DOCUMENT_REFERENCE_COUNT == 109
    assert observed_payload_hashes == expected_payload_hashes
    assert len(digest_by_document) == EXPECTED_DOCUMENT_COUNT == 60


def test_task_024_exercises_all_four_decomposition_forms(frozen_inputs) -> None:
    task = next(task for task in frozen_inputs.tasks if task.id == "task_024")
    agent = _agent(frozen_inputs, task, "inplace_image")
    documents = agent.runtime.documents

    assert [document.decomposition_form for document in documents] == [
        "P_B",
        "B",
        "B",
        "P_B_S",
        "B_S",
        "B",
    ]
    assert {document.decomposition_form for document in documents} == {
        "B",
        "B_S",
        "P_B",
        "P_B_S",
    }

    for document in documents:
        derived_form = (
            ("P_" if document.prefix else "") + "B" + ("_S" if document.suffix else "")
        )
        assert document.decomposition_form == derived_form
        assert document.prefix + document.block + document.suffix == (
            document.source_content
        )
        assert (
            document.source_content[document.char_start : document.char_end]
            == document.block
        )

        heading = f"## {document.title}\n\n"
        section = heading + document.source_content
        parts = build_inplace_system_content(
            system_prompt=section,
            domain_policy=section,
            documents=(document,),
        )
        expected_types = ["text", "image_url"]
        if document.suffix:
            expected_types.append("text")
        assert [part["type"] for part in parts] == expected_types
        assert parts[0] == {"type": "text", "text": heading + document.prefix}
        assert hashlib.sha256(_decode_inline_png(parts[1])).hexdigest() == (
            document.image_sha256
        )
        if document.suffix:
            assert parts[2] == {"type": "text", "text": document.suffix}


def test_user_history_is_never_used_as_an_image_injection_channel(
    frozen_inputs,
) -> None:
    task = next(task for task in frozen_inputs.tasks if task.id == "task_024")
    history = [
        UserMessage(role="user", content="first caller message"),
        UserMessage(role="user", content="second caller message"),
    ]
    before = [message.model_dump() for message in history]

    for arm in ARMS:
        state = _agent(frozen_inputs, task, arm).get_init_state(history)
        assert state.messages == history
        assert state.messages is not history
        assert len(state.messages) == len(history)
        assert all(message.image_pages is None for message in state.messages)
        wire = to_litellm_messages(state.system_messages + state.messages)
        assert [message["role"] for message in wire] == [
            "system",
            "user",
            "user",
        ]

    assert [message.model_dump() for message in history] == before


def test_registered_runner_entry_builds_only_frozen_tasks(
    frozen_inputs, monkeypatch
) -> None:
    """Exercise the same registry/build_agent entry used by Tau2 runs."""
    agent_name = "llm_agent_complementary_inplace_system_v2"
    task = next(task for task in frozen_inputs.tasks if task.id == "task_024")
    environment = get_environment(
        retrieval_variant="golden_retrieval",
        task=task,
    )
    monkeypatch.setenv(ARM_ENV, "inplace_image")
    agent = build_agent(
        agent_name,
        environment,
        llm="test-model",
        llm_args={},
        task=task,
    )
    assert isinstance(agent, ComplementaryInplaceSystemAgent)
    assert agent.runtime.arm == "inplace_image"
    assert len(agent.runtime.documents) == 6

    task_filter = registry.get_agent_task_filter(agent_name)
    assert task_filter is not None
    assert [
        candidate.id for candidate in get_tasks() if task_filter(candidate)
    ] == frozen_inputs.selection["task_ids"]


def test_to_litellm_preserves_ordered_multimodal_system_content() -> None:
    parts = [
        {"type": "text", "text": "A"},
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,Zmlyc3Q="},
        },
        {"type": "text", "text": "S"},
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,c2Vjb25k"},
        },
        {"type": "text", "text": "C"},
    ]
    message = SystemMessage(role="system", content=parts)

    validate_message(message)
    converted = to_litellm_messages([message])

    assert converted == [{"role": "system", "content": parts}]
    assert [part["type"] for part in converted[0]["content"]] == [
        "text",
        "image_url",
        "text",
        "image_url",
        "text",
    ]


def test_projection_snapshots_contain_hashes_and_sizes_but_no_base64(
    frozen_inputs,
) -> None:
    task = next(task for task in frozen_inputs.tasks if task.id == "task_024")
    image_agent = _agent(frozen_inputs, task, "inplace_image")
    image_snapshot = image_agent.request_projection_snapshot()
    serialized = json.dumps(image_snapshot, ensure_ascii=False, sort_keys=True)

    assert image_snapshot["arm"] == "inplace_image"
    assert image_snapshot["injected_user_message_count"] == 0
    assert image_snapshot["document_ids"] == [
        document.document_id for document in image_agent.runtime.documents
    ]
    assert image_snapshot["document_image_sha256"] == [
        document.image_sha256 for document in image_agent.runtime.documents
    ]
    assert image_snapshot["system"]["content_kind"] == "multimodal"
    system_image_parts = [
        part
        for part in image_snapshot["system"]["parts"]
        if part["type"] == "image_url"
    ]
    assert [part["sha256"] for part in system_image_parts] == (
        image_snapshot["document_image_sha256"]
    )
    assert all(part["byte_count"] > 0 for part in system_image_parts)
    assert "base64" not in serialized
    assert "data:image" not in serialized

    text_agent = _agent(frozen_inputs, task, "inplace_text")
    text_snapshot = text_agent.request_projection_snapshot()
    assert text_snapshot["system"] == {
        "content_kind": "text",
        "text_char_count": len(text_agent.system_prompt),
        "text_sha256": hashlib.sha256(
            text_agent.system_prompt.encode("utf-8")
        ).hexdigest(),
        "parts": [],
    }
    assert "base64" not in json.dumps(text_snapshot, sort_keys=True)


@pytest.mark.parametrize("arm", ["full_text", "complementary", "unknown"])
def test_invalid_arm_is_rejected(frozen_inputs, arm: str) -> None:
    task = frozen_inputs.tasks[0]
    with pytest.raises(ComplementaryInplaceSystemError, match="must be one of"):
        _agent(frozen_inputs, task, arm)


@pytest.mark.parametrize(
    "content",
    [
        None,
        "   ",
        [],
        [{"type": "text", "text": ""}],
        [{"type": "image_url", "image_url": {}}],
        [{"type": "image_url", "image_url": "not-an-object"}],
        [{"type": "unsupported", "value": "x"}],
    ],
)
def test_system_message_validation_rejects_malformed_content(content) -> None:
    with pytest.raises(AssertionError):
        validate_message(SystemMessage(role="system", content=content))


def test_system_message_validation_and_displays_redact_inline_bytes() -> None:
    secret = "secret-inline-image-payload-must-not-leak"
    message = SystemMessage(
        role="system",
        content=[
            {"type": "text", "text": "visible prefix"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{secret}"},
            },
            {"type": "text", "text": "visible suffix"},
        ],
    )

    validate_message(message)
    validate_message(SystemMessage(role="system", content="plain system text"))
    assert message.has_content()
    renderings = [
        message.display_content(),
        str(message),
        MarkdownDisplay.display_message(message),
    ]
    for rendered in renderings:
        assert "visible prefix" in rendered
        assert "visible suffix" in rendered
        assert "[System image; inline bytes omitted]" in rendered
        assert secret not in rendered
        assert "data:image" not in rendered
    assert litellm.redact_messages_in_exceptions is True
