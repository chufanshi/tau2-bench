"""Agent for the boundary-refined v3 in-place system-image condition."""

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
from tau2.domains.banking_knowledge.complementary_inplace_system_v3 import (
    EXPECTED_MANIFEST_SHA256,
    FROZEN_TASK_IDS,
    ComplementaryInplaceSystemV3Error,
    RuntimeTask,
    load_task_runtime,
)
from tau2.environment.tool import Tool


class ComplementaryInplaceSystemAgentV3(LLMAgent):
    """Matched text/image agent over the frozen v3 boundary overlay."""

    @classmethod
    def check_valid_task(cls, task: Task) -> bool:
        return task.id in FROZEN_TASK_IDS

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
        super().__init__(tools=tools, domain_policy=domain_policy, llm=llm, llm_args=llm_args)
        if task is None:
            raise ComplementaryInplaceSystemV3Error("v3 agent requires the current task")
        self.runtime: RuntimeTask = load_task_runtime(
            task=task,
            domain_policy=domain_policy,
            arm=arm,
            manifest_path=manifest_path,
            asset_root=asset_root,
        )

    def get_init_state(self, message_history: Optional[list[Message]] = None) -> LLMAgentStateType:
        state = super().get_init_state(
            message_history=list(message_history) if message_history is not None else None
        )
        if self.runtime.arm == "inplace_text":
            return state
        content = build_inplace_system_content(
            system_prompt=self.system_prompt,
            domain_policy=self.runtime.domain_policy,
            documents=self.runtime.documents,
        )
        state.system_messages = [SystemMessage(role="system", content=content)]
        return state

    def request_projection_snapshot(self) -> dict:
        state = self.get_init_state()
        content = state.system_messages[0].content
        if content is None:
            raise ComplementaryInplaceSystemV3Error("system projection is empty")
        return {
            "task_id": self.runtime.task_id,
            "arm": self.runtime.arm,
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "document_ids": [doc.document_id for doc in self.runtime.documents],
            "document_image_sha256": [doc.image_sha256 for doc in self.runtime.documents],
            "system": system_content_snapshot(content),
            "injected_user_message_count": 0,
        }


def create_complementary_inplace_system_agent_v3(tools, domain_policy, **kwargs):
    task = kwargs.get("task")
    if task is None:
        raise ComplementaryInplaceSystemV3Error(
            "llm_agent_complementary_inplace_system_v3 requires a task"
        )
    return ComplementaryInplaceSystemAgentV3(
        tools=tools,
        domain_policy=domain_policy,
        task=task,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
    )
