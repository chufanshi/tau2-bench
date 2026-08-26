"""Matched three-arm agent for the frozen selected complementary runtime."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Optional

from tau2.agent.llm_agent import LLMAgent, LLMAgentStateType
from tau2.data_model.message import Message, UserMessage
from tau2.data_model.tasks import Task
from tau2.domains.banking_knowledge.selected_complementary import (
    SelectedComplementaryRuntimeError,
    load_task_runtime,
)
from tau2.environment.tool import Tool

MATCHED_SUPPLEMENT_ENVELOPE = (
    "This private required-document supplement carries the remaining "
    "authoritative facts. Combine it with the matching F_T entries in the "
    "system policy."
)


class SelectedComplementaryAgent(LLMAgent):
    """Agent whose private initial document messages are arm-matched.

    ``full_text`` receives no private document message because its complete
    source text is already in the frozen system policy. ``partitioned_text``
    and ``complementary`` share the identical ``F_T`` system policy and the
    same ordered bridge messages. The former adds the frozen ``F_I`` textual
    projection; the latter adds the reviewed PNGs covering exactly ``F_I``.
    """

    def __init__(
        self,
        tools: list[Tool],
        domain_policy: str,
        task: Task,
        llm: str,
        llm_args: Optional[dict] = None,
        dataset_path: Optional[Path] = None,
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
            raise SelectedComplementaryRuntimeError(
                "SelectedComplementaryAgent requires the current task"
            )
        self.runtime = load_task_runtime(
            task=task,
            domain_policy=domain_policy,
            arm=arm,
            dataset_path=dataset_path,
            asset_root=asset_root,
        )

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> LLMAgentStateType:
        private_history = list(message_history) if message_history is not None else None
        state = super().get_init_state(message_history=private_history)
        if self.runtime.arm == "full_text":
            return state

        document_messages: list[UserMessage] = [
            UserMessage(
                role="user",
                content=MATCHED_SUPPLEMENT_ENVELOPE,
            )
        ]
        for document in self.runtime.documents:
            if self.runtime.arm == "partitioned_text":
                content = f"{document.bridge_text}\n\n{document.fi_text}"
                image_pages = None
            else:
                content = document.bridge_text
                image_pages = [
                    base64.b64encode(payload).decode("ascii")
                    for payload in document.image_payloads
                ]
            document_messages.append(
                UserMessage(
                    role="user",
                    content=content,
                    image_pages=image_pages,
                )
            )

        # Insert as one ordered prefix without mutating caller-owned history.
        state.messages[0:0] = document_messages
        return state


def create_selected_complementary_agent(tools, domain_policy, **kwargs):
    """Factory for the frozen selected complementary v1 agent."""
    task = kwargs.get("task")
    if task is None:
        raise SelectedComplementaryRuntimeError(
            "llm_agent_selected_complementary_v1 requires a task"
        )
    return SelectedComplementaryAgent(
        tools=tools,
        domain_policy=domain_policy,
        task=task,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
    )
