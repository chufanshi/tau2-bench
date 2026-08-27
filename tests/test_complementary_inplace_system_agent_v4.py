"""Model-free tests for the source-native v4 paired system agent."""

from __future__ import annotations

import base64
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

import tau2.agent.complementary_inplace_system_agent_v4 as agent_module
import tau2.domains.banking_knowledge.complementary_inplace_system_v4 as runtime_module
from tau2.agent.complementary_inplace_system_agent_v4 import (
    ComplementaryInplaceSystemAgentV4,
)
from tau2.data_model.message import UserMessage
from tau2.domains.banking_knowledge.complementary_inplace_system_v4 import (
    ComplementaryInplaceSystemV4Error,
    RuntimeDocument,
    RuntimeTask,
)
from tau2.registry import registry

WORKSPACE = Path(__file__).resolve().parents[2]
PNG_PREFIX = "data:image/png;base64,"


def _png_header(width: int = 7, height: int = 5) -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n"
        + struct.pack(">I", 13)
        + b"IHDR"
        + struct.pack(">II", width, height)
    )


def _runtime(arm: str) -> RuntimeTask:
    source = "Retained prefix.\n\nSELECTED POLICY BLOCK\n\nRetained suffix."
    block = "SELECTED POLICY BLOCK"
    start = source.index(block)
    document = RuntimeDocument(
        document_id="doc_fixture_001",
        document_key="fixture_001",
        title="Fixture policy",
        source_content=source,
        prefix=source[:start],
        block=block,
        suffix=source[start + len(block) :],
        char_start=start,
        char_end=start + len(block),
        decomposition_form="P_B_S",
        image_payload=_png_header(),
        image_sha256="b" * 64,
        image_width=7,
        image_height=5,
        image_path=WORKSPACE / "fixture.png",
    )
    policy = f"Golden header\n\n## {document.title}\n\n{source}\n\nGolden footer"
    return RuntimeTask(
        arm=arm,
        task_id="task_fixture",
        domain_policy=policy,
        documents=(document,),
        manifest_path=WORKSPACE / "fixture-manifest.json",
        asset_root=WORKSPACE,
        manifest_sha256="a" * 64,
    )


def _agent(
    monkeypatch: pytest.MonkeyPatch, arm: str
) -> ComplementaryInplaceSystemAgentV4:
    runtime = _runtime(arm)
    monkeypatch.setattr(agent_module, "load_task_runtime", lambda **_kwargs: runtime)
    return ComplementaryInplaceSystemAgentV4(
        tools=[],
        domain_policy=runtime.domain_policy,
        task=SimpleNamespace(id=runtime.task_id),
        llm="never-called-model",
        arm=arm,
    )


def test_v4_registry_name_is_present_without_loading_assets() -> None:
    assert registry.get_agent_factory("llm_agent_complementary_inplace_system_v4")
    assert registry.get_agent_task_filter("llm_agent_complementary_inplace_system_v4")


def test_v4_text_and_image_arms_use_one_system_message_and_no_user_injection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history = [UserMessage(role="user", content="existing user history")]
    text_agent = _agent(monkeypatch, "inplace_text")
    text_state = text_agent.get_init_state(history)
    assert len(text_state.system_messages) == 1
    text_content = text_state.system_messages[0].content
    assert isinstance(text_content, str)
    assert text_content == text_agent.system_prompt
    assert text_state.messages == history

    image_agent = _agent(monkeypatch, "inplace_image")
    image_state = image_agent.get_init_state(history)
    assert len(image_state.system_messages) == 1
    parts = image_state.system_messages[0].content
    assert isinstance(parts, list)
    assert image_state.messages == history
    assert [part["type"] for part in parts] == ["text", "image_url", "text"]
    reconstructed = []
    for part in parts:
        if part["type"] == "text":
            reconstructed.append(part["text"])
        else:
            url = part["image_url"]["url"]
            assert url.startswith(PNG_PREFIX)
            assert (
                base64.b64decode(url[len(PNG_PREFIX) :], validate=True) == _png_header()
            )
            reconstructed.append(image_agent.runtime.documents[0].block)
    assert "".join(reconstructed) == text_content
    snapshot = image_agent.request_projection_snapshot()
    assert snapshot["system_message_count"] == 1
    assert snapshot["injected_user_message_count"] == 0
    assert snapshot["system"]["content_kind"] == "multimodal"


def test_v4_unpinned_or_missing_manifest_fails_before_any_model_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runtime_module,
        "EXPECTED_MANIFEST_SHA256",
        "PENDING_218_PIXEL_PASS_AND_FINAL_MANIFEST_FREEZE",
    )
    with pytest.raises(ComplementaryInplaceSystemV4Error, match="not released"):
        runtime_module.expected_manifest_sha256()

    monkeypatch.setattr(runtime_module, "EXPECTED_MANIFEST_SHA256", "0" * 64)
    missing = WORKSPACE / "tauvision/data/definitely-missing-v4-manifest.json"
    with pytest.raises(ComplementaryInplaceSystemV4Error, match="missing"):
        runtime_module.load_manifest_metadata(missing, WORKSPACE)
