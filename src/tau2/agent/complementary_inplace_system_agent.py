"""Agent for byte-controlled in-place images inside the system policy."""

from __future__ import annotations

import base64
import binascii
import hashlib
import struct
from pathlib import Path
from typing import Any, Optional

from tau2.agent.llm_agent import LLMAgent, LLMAgentStateType
from tau2.data_model.message import Message, SystemContent, SystemMessage
from tau2.data_model.tasks import Task
from tau2.domains.banking_knowledge.complementary_inplace_system import (
    EXPECTED_MANIFEST_SHA256,
    FROZEN_TASK_IDS,
    ComplementaryInplaceSystemError,
    RuntimeDocument,
    RuntimeTask,
    load_task_runtime,
)
from tau2.environment.tool import Tool


def _append_text(parts: list[dict[str, Any]], text: str) -> None:
    """Append exact non-empty text, merging only adjacent text parts."""
    if text == "":
        return
    if parts and parts[-1].get("type") == "text":
        parts[-1]["text"] += text
    else:
        parts.append({"type": "text", "text": text})


def build_inplace_system_content(
    *,
    system_prompt: str,
    domain_policy: str,
    documents: tuple[RuntimeDocument, ...],
) -> list[dict[str, Any]]:
    """Replace each document's exact B span with its reviewed PNG.

    All coordinates are derived from a uniquely located full document section,
    never from a global replacement of B itself.  Therefore repeated policy
    phrases cannot move an image to the wrong source occurrence.
    """
    if system_prompt.count(domain_policy) != 1:
        raise ComplementaryInplaceSystemError(
            "Golden domain policy must occur exactly once in the system prompt"
        )
    policy_start = system_prompt.index(domain_policy)
    spans: list[tuple[int, int, RuntimeDocument]] = []
    previous_section_end = -1
    for document in documents:
        heading = f"## {document.title}\n\n"
        section = heading + document.source_content
        if domain_policy.count(section) != 1:
            raise ComplementaryInplaceSystemError(
                f"task document {document.document_id!r} must occur exactly once "
                "in the Golden policy"
            )
        section_start = domain_policy.index(section)
        section_end = section_start + len(section)
        if section_start <= previous_section_end:
            raise ComplementaryInplaceSystemError(
                f"document {document.document_id!r} is out of required-document order"
            )
        previous_section_end = section_end
        content_start = policy_start + section_start + len(heading)
        block_start = content_start + document.char_start
        block_end = content_start + document.char_end
        if system_prompt[block_start:block_end] != document.block:
            raise ComplementaryInplaceSystemError(
                f"document {document.document_id!r} block moved before serialization"
            )
        if block_start >= block_end:
            raise ComplementaryInplaceSystemError(
                f"document {document.document_id!r} has an empty replacement span"
            )
        spans.append((block_start, block_end, document))

    parts: list[dict[str, Any]] = []
    cursor = 0
    for block_start, block_end, document in spans:
        if block_start < cursor:
            raise ComplementaryInplaceSystemError(
                f"document {document.document_id!r} overlaps a prior replacement"
            )
        _append_text(parts, system_prompt[cursor:block_start])
        encoded = base64.b64encode(document.image_payload).decode("ascii")
        parts.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{encoded}",
                },
            }
        )
        cursor = block_end
    _append_text(parts, system_prompt[cursor:])
    if not parts or not any(part.get("type") == "image_url" for part in parts):
        raise ComplementaryInplaceSystemError(
            "inplace_image serialization produced no image parts"
        )
    if any(part.get("type") == "text" and part.get("text") == "" for part in parts):
        raise ComplementaryInplaceSystemError(
            "inplace_image serialization produced an empty text part"
        )
    return parts


def system_content_snapshot(content: SystemContent) -> dict[str, Any]:
    """Return auditable hashes and sizes without retaining inline base64 bytes."""
    if isinstance(content, str):
        return {
            "content_kind": "text",
            "text_char_count": len(content),
            "text_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "parts": [],
        }

    snapshot_parts: list[dict[str, Any]] = []
    for index, part in enumerate(content):
        part_type = part.get("type")
        if part_type == "text":
            text = part.get("text")
            if not isinstance(text, str):
                raise ComplementaryInplaceSystemError(
                    f"system text part {index} is malformed"
                )
            snapshot_parts.append(
                {
                    "index": index,
                    "type": "text",
                    "char_count": len(text),
                    "byte_count": len(text.encode("utf-8")),
                    "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                }
            )
            continue
        if part_type != "image_url":
            raise ComplementaryInplaceSystemError(
                f"system content part {index} has unsupported type {part_type!r}"
            )
        image_url = part.get("image_url")
        if not isinstance(image_url, dict):
            raise ComplementaryInplaceSystemError(
                f"system image part {index} has no image_url object"
            )
        url = image_url.get("url")
        prefix = "data:image/png;base64,"
        if not isinstance(url, str) or not url.startswith(prefix):
            raise ComplementaryInplaceSystemError(
                f"system image part {index} is not an inline PNG"
            )
        try:
            payload = base64.b64decode(url[len(prefix) :], validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ComplementaryInplaceSystemError(
                f"system image part {index} has invalid base64"
            ) from exc
        if (
            len(payload) < 24
            or payload[:8] != b"\x89PNG\r\n\x1a\n"
            or payload[12:16] != b"IHDR"
        ):
            raise ComplementaryInplaceSystemError(
                f"system image part {index} is not a canonical PNG"
            )
        width, height = struct.unpack(">II", payload[16:24])
        snapshot_parts.append(
            {
                "index": index,
                "type": "image_url",
                "mime_type": "image/png",
                "byte_count": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "width": width,
                "height": height,
            }
        )
    return {
        "content_kind": "multimodal",
        "part_count": len(content),
        "parts": snapshot_parts,
    }


class ComplementaryInplaceSystemAgent(LLMAgent):
    """Two-arm agent whose image condition changes only system-policy B spans."""

    @classmethod
    def check_valid_task(cls, task: Task) -> bool:
        """Return whether a task belongs to the frozen 5-of-5 cohort."""
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
        super().__init__(
            tools=tools,
            domain_policy=domain_policy,
            llm=llm,
            llm_args=llm_args,
        )
        if task is None:
            raise ComplementaryInplaceSystemError(
                "ComplementaryInplaceSystemAgent requires the current task"
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
        """Create one system message without adding or changing user messages."""
        copied_history = list(message_history) if message_history is not None else None
        state = super().get_init_state(message_history=copied_history)
        if self.runtime.arm == "inplace_text":
            return state
        content = build_inplace_system_content(
            system_prompt=self.system_prompt,
            domain_policy=self.runtime.domain_policy,
            documents=self.runtime.documents,
        )
        state.system_messages = [SystemMessage(role="system", content=content)]
        return state

    def request_projection_snapshot(self) -> dict[str, Any]:
        """Return the first-turn system projection without exposing image bytes."""
        state = self.get_init_state()
        content = state.system_messages[0].content
        if content is None:
            raise ComplementaryInplaceSystemError("system projection is empty")
        return {
            "task_id": self.runtime.task_id,
            "arm": self.runtime.arm,
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "document_ids": [doc.document_id for doc in self.runtime.documents],
            "document_image_sha256": [
                doc.image_sha256 for doc in self.runtime.documents
            ],
            "system": system_content_snapshot(content),
            "injected_user_message_count": 0,
        }


def create_complementary_inplace_system_agent(tools, domain_policy, **kwargs):
    """Factory for the frozen EXP4 in-place system-policy v2 agent."""
    task = kwargs.get("task")
    if task is None:
        raise ComplementaryInplaceSystemError(
            "llm_agent_complementary_inplace_system_v2 requires a task"
        )
    return ComplementaryInplaceSystemAgent(
        tools=tools,
        domain_policy=domain_policy,
        task=task,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
    )
