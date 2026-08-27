"""Model-free regression tests for the boundary-refined v3 paired agent."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

from tau2.agent.complementary_inplace_system_agent_v3 import (
    ComplementaryInplaceSystemAgentV3,
)
from tau2.agent.llm_agent import LLMAgent
from tau2.data_model.message import UserMessage
from tau2.domains.banking_knowledge.complementary_inplace_system_v3 import (
    ARMS,
    EXPECTED_FORMS,
    EXPECTED_MANIFEST_SHA256,
    FROZEN_TASK_IDS,
)
from tau2.domains.banking_knowledge.environment import get_knowledge_base, get_tasks
from tau2.domains.banking_knowledge.retrieval import PROMPTS_DIR, golden_prompt
from tau2.registry import registry


WORKSPACE = Path(__file__).resolve().parents[2]
MANIFEST = WORKSPACE / "tauvision/data/banking_explorer/complementary_inplace_system_v3/asset_manifest.json"
SELECTION = WORKSPACE / "tauvision/data/banking_explorer/complementary_5of5_selection_v1.json"
PNG_PREFIX = "data:image/png;base64,"


def _inputs():
    selection = json.loads(SELECTION.read_text())
    tasks = {task.id: task for task in get_tasks()}
    kb = get_knowledge_base()
    return selection, tasks, kb


def _agent(task, policy: str, arm: str) -> ComplementaryInplaceSystemAgentV3:
    return ComplementaryInplaceSystemAgentV3(
        tools=[], domain_policy=policy, task=task, llm="test-model",
        manifest_path=MANIFEST, asset_root=WORKSPACE, arm=arm,
    )


def test_v3_manifest_and_registry_are_frozen() -> None:
    assert hashlib.sha256(MANIFEST.read_bytes()).hexdigest() == EXPECTED_MANIFEST_SHA256
    manifest = json.loads(MANIFEST.read_text())
    assert manifest["counts"]["forms"] == EXPECTED_FORMS
    assert manifest["counts"]["origins"] == {"generated_v3": 16, "reused_exact_v2": 44}
    assert registry.get_agent_task_filter("llm_agent_complementary_inplace_system_v3")


def test_all_19_tasks_preserve_text_and_reconstruct_109_image_replacements() -> None:
    selection, tasks, kb = _inputs()
    assert tuple(selection["task_ids"]) == FROZEN_TASK_IDS
    observed_refs = 0
    forms = set()
    digest_by_document = {}
    for task_id in selection["task_ids"]:
        task = tasks[task_id]
        policy = golden_prompt(PROMPTS_DIR / "required_docs.md", kb, task)
        history = [UserMessage(role="user", content=f"history {task_id}")]
        baseline = LLMAgent(tools=[], domain_policy=policy, llm="test-model").get_init_state(history)
        text_agent = _agent(task, policy, "inplace_text")
        text_state = text_agent.get_init_state(history)
        assert text_state.system_messages == baseline.system_messages
        assert isinstance(text_state.system_messages[0].content, str)
        image_agent = _agent(task, policy, "inplace_image")
        image_state = image_agent.get_init_state(history)
        content = image_state.system_messages[0].content
        assert isinstance(content, list) and image_state.messages == history
        reconstructed = []
        image_index = 0
        for part in content:
            if part["type"] == "text":
                assert part["text"]
                reconstructed.append(part["text"])
            else:
                document = image_agent.runtime.documents[image_index]
                url = part["image_url"]["url"]
                assert url.startswith(PNG_PREFIX)
                payload = base64.b64decode(url[len(PNG_PREFIX):], validate=True)
                digest = hashlib.sha256(payload).hexdigest()
                assert digest == document.image_sha256
                assert digest_by_document.setdefault(document.document_id, digest) == digest
                reconstructed.append(document.block)
                forms.add(document.decomposition_form)
                observed_refs += 1
                image_index += 1
        assert image_index == len(image_agent.runtime.documents)
        assert "".join(reconstructed) == text_state.system_messages[0].content
        assert image_agent.request_projection_snapshot()["injected_user_message_count"] == 0
    assert observed_refs == 109
    assert len(digest_by_document) == 60
    assert forms == set(EXPECTED_FORMS)
    assert ARMS == ("inplace_text", "inplace_image")
