"""Thin paired agent for the source-native v4 system-message release."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from tau2.agent.complementary_inplace_system_agent import (
    build_inplace_system_content,
    system_content_snapshot,
)
from tau2.agent.llm_agent import LLMAgent, LLMAgentStateType
from tau2.data_model.message import Message, SystemMessage
from tau2.data_model.tasks import Task
from tau2.domains.banking_knowledge.complementary_inplace_system_v4 import (
    ComplementaryInplaceSystemV4Error,
    RuntimeTask,
    is_frozen_task,
    load_task_runtime,
)
from tau2.environment.tool import Tool


class ComplementaryInplaceSystemAgentV4(LLMAgent):
    """Keep the text arm unchanged and replace only exact B spans in-image."""

    @classmethod
    def check_valid_task(cls, task: Task) -> bool:
        """Read task membership from the hash-pinned runtime manifest."""

        return is_frozen_task(task)

    def __init__(
        self,
        tools: list[Tool],
        domain_policy: str,
        task: Task,
        llm: str,
        llm_args: Optional[dict] = None,
        manifest_path: Optional[Path] = None,
        asset_root: Optional[Path] = None,
        arm: Optional[str] = None,
    ):
        super().__init__(
            tools=tools,
            domain_policy=domain_policy,
            llm=llm,
            llm_args=llm_args,
        )
        if task is None:
            raise ComplementaryInplaceSystemV4Error(
                "v4 agent requires the current task"
            )
        self.runtime: RuntimeTask = load_task_runtime(
            task=task,
            domain_policy=domain_policy,
            arm=arm,
            manifest_path=manifest_path,
            asset_root=asset_root,
        )

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> LLMAgentStateType:
        """Create exactly one system message and never inject a user message."""

        history = list(message_history) if message_history is not None else None
        state = super().get_init_state(message_history=history)
        if len(state.system_messages) != 1:
            raise ComplementaryInplaceSystemV4Error(
                "v4 must serialize exactly one system message"
            )
        if self.runtime.arm == "inplace_text":
            content = state.system_messages[0].content
            if content != self.system_prompt or not isinstance(content, str):
                raise ComplementaryInplaceSystemV4Error(
                    "inplace_text must be the original single Text(D) system prompt"
                )
            return state
        content = build_inplace_system_content(
            system_prompt=self.system_prompt,
            domain_policy=self.runtime.domain_policy,
            documents=self.runtime.documents,
        )
        state.system_messages = [SystemMessage(role="system", content=content)]
        return state

    def request_projection_snapshot(self) -> dict:
        """Return hashes and dimensions, never inline image bytes."""

        state = self.get_init_state()
        if len(state.system_messages) != 1:
            raise ComplementaryInplaceSystemV4Error(
                "v4 request projection has multiple system messages"
            )
        content = state.system_messages[0].content
        if content is None:
            raise ComplementaryInplaceSystemV4Error("system projection is empty")
        return {
            "task_id": self.runtime.task_id,
            "arm": self.runtime.arm,
            "manifest_sha256": self.runtime.manifest_sha256,
            "document_ids": [doc.document_id for doc in self.runtime.documents],
            "document_image_sha256": [
                doc.image_sha256 for doc in self.runtime.documents
            ],
            "system_message_count": 1,
            "system": system_content_snapshot(content),
            "injected_user_message_count": 0,
        }


def create_complementary_inplace_system_agent_v4(tools, domain_policy, **kwargs):
    """Registry factory for the source-native v4 paired agent."""

    task = kwargs.get("task")
    if task is None:
        raise ComplementaryInplaceSystemV4Error(
            "llm_agent_complementary_inplace_system_v4 requires a task"
        )
    return ComplementaryInplaceSystemAgentV4(
        tools=tools,
        domain_policy=domain_policy,
        task=task,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
    )
