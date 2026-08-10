"""Deterministic image-trigger resolution without exposing symbolic kappa."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from tau2.data_model.tasks import Task


@dataclass(frozen=True)
class RuntimeImage:
    """One frozen asset selected for a user-message attachment."""

    trigger_index: int
    kappa_index: int
    asset_id: str
    image_content: str
    alt_text: str


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _condition_matches(condition: Any, agent_text: str | None) -> bool:
    """Match the small, release-time trigger language deterministically."""

    if condition is None:
        return True
    if not agent_text:
        return False
    text = re.sub(r"\s+", " ", agent_text.casefold()).strip()

    if isinstance(condition, dict):
        if condition.get("type") != "agent_request_fields":
            raise ValueError(f"unsupported image trigger condition: {condition!r}")
        fields = condition.get("fields")
        if not isinstance(fields, list) or not fields:
            raise ValueError(f"invalid agent_request_fields condition: {condition!r}")
        requirements = set(fields)
    elif isinstance(condition, str):
        declared = condition.casefold()
        requirements: set[str] = set()
        if "serial" in declared:
            requirements.add("serial_plate")
        if "close-up" in declared and "damage" in declared:
            requirements.add("damage_close_up")
        if not requirements:
            raise ValueError(f"unsupported image trigger condition: {condition!r}")
    else:
        raise ValueError(f"unsupported image trigger condition: {condition!r}")

    for requirement in requirements:
        if requirement == "serial_plate":
            if "serial" not in text:
                return False
        elif requirement == "damage_close_up":
            has_damage = _contains_any(
                text, ("damage", "damaged", "burn", "defect", "heating element")
            )
            has_image_request = _contains_any(
                text,
                (
                    "close-up",
                    "close up",
                    "closer",
                    "clear photo",
                    "clearer photo",
                    "clear picture",
                    "clearer picture",
                    "clear image",
                    "clearer image",
                ),
            )
            if not (has_damage and has_image_request):
                return False
        else:
            raise ValueError(f"unsupported image trigger field: {requirement!r}")
    return True


def resolve_image_trigger(
    *,
    task: Task,
    event: str,
    agent_text: str | None,
    image_seed: int,
    fired_trigger_indices: set[int],
) -> RuntimeImage | None:
    """Resolve at most one unfired trigger for the current user turn."""

    matches: list[RuntimeImage] = []
    assets = task.runtime_image_assets or {}
    for trigger_index, trigger in enumerate(task.image_triggers or []):
        if trigger_index in fired_trigger_indices or trigger.get("when") != event:
            continue
        if event == "on_request" and not _condition_matches(
            trigger.get("condition"), agent_text
        ):
            continue
        if event not in {"opening", "on_request"}:
            raise ValueError(f"unsupported image trigger event: {event!r}")
        kappa_index = trigger.get("kappa_index")
        asset_ids = trigger.get("asset_ids")
        if type(kappa_index) is not int or not isinstance(asset_ids, list) or not asset_ids:
            raise ValueError(
                f"task {task.id} trigger {trigger_index} has no published assets"
            )
        asset_id = asset_ids[image_seed % len(asset_ids)]
        try:
            record = assets[asset_id]
            image_content = record["image_content"]
        except KeyError as exc:
            raise ValueError(
                f"task {task.id} trigger references unloaded asset {asset_id}"
            ) from exc
        matches.append(
            RuntimeImage(
                trigger_index=trigger_index,
                kappa_index=kappa_index,
                asset_id=asset_id,
                image_content=image_content,
                alt_text=record.get(
                    "alt_text", "User-submitted product evidence photo."
                ),
            )
        )
    if len(matches) > 1:
        raise ValueError(
            f"task {task.id} fires multiple images in one user turn; "
            "the message schema supports one attachment"
        )
    return matches[0] if matches else None
