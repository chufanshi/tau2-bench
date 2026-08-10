"""Executable grounding and deterministic scoring for Track A pilot tasks."""

import hashlib
import json

import pytest

from tau2.data_model.message import (
    AssistantMessage,
    SystemMessage,
    Tick,
    ToolCall,
    UserMessage,
)
from tau2.domains.retail_vision.environment import (
    get_environment,
    get_released_tasks,
    get_tasks,
)
from tau2.evaluator.evaluator_action import (
    ActionEvaluator,
    FullDuplexActionEvaluator,
)
from tau2.evaluator.evaluator_env import EnvironmentEvaluator
from tau2.utils.llm_utils import to_litellm_messages
from tau2.user.user_simulator_base import UserState
from tau2.vision.runtime import resolve_image_trigger


EXPECTED_RESOLUTIONS = {
    "refund_order": "full_refund",
    "create_replacement": "replacement",
    "issue_voucher": "shipping_voucher_15",
    "create_ticket": "manual_review",
}


@pytest.mark.parametrize("task", get_tasks("base"), ids=lambda task: task.id)
def test_grounded_task_golden_action_replays_to_unique_db_state(task):
    action = task.evaluation_criteria.actions[-1]
    call = ToolCall(
        id=f"call-{task.id}",
        name=action.name,
        arguments=action.arguments,
        requestor="assistant",
    )
    live = get_environment()
    result = live.get_response(call)
    assert not result.error
    claim = next(iter(live.tools.db.visual_claims.values()))
    assert claim.resolution == EXPECTED_RESOLUTIONS[action.name]

    trajectory = [
        AssistantMessage(role="assistant", content=None, tool_calls=[call]),
        result,
    ]
    env_reward = EnvironmentEvaluator.calculate_reward(
        environment_constructor=get_environment,
        task=task,
        full_trajectory=trajectory,
    )
    assert env_reward.reward == 1.0
    action_reward = ActionEvaluator.calculate_reward(task, trajectory)
    assert action_reward.reward == 1.0
    assert action_reward.info["forbidden_actions_called"] == []


def test_forbidden_remedy_fails_half_and_full_duplex_action_scoring():
    task = get_tasks("base")[0]
    forbidden = task.evaluation_criteria.forbidden_actions[0]
    call = ToolCall(id="wrong", name=forbidden, arguments={})
    half_trajectory = [
        AssistantMessage(role="assistant", content=None, tool_calls=[call])
    ]
    half = ActionEvaluator.calculate_reward(task, half_trajectory)
    assert half.reward == 0.0
    assert half.info["forbidden_actions_called"] == [forbidden]

    tick = Tick(
        tick_id=0,
        timestamp="2026-01-01T00:00:00Z",
        agent_tool_calls=[call],
    )
    full = FullDuplexActionEvaluator.calculate_reward(task, [tick])
    assert full.reward == 0.0
    assert full.info["forbidden_actions_called"] == [forbidden]


def test_visual_task_fields_survive_tau2_validation():
    task = get_tasks("base")[0]
    assert task.kappa_refs
    assert task.image_triggers
    assert task.sim_knowledge
    assert task.annotations["grounding"]["db_sha256"]
    grounding = task.annotations["grounding"]
    assert grounding["instance_serial"] == f"TV-{grounding['item_id']}"


def test_instance_serials_are_deterministic_database_state():
    db = get_environment().tools.db
    assert db.instance_serials["#W8587412:9747045638"] == "TV-9747045638"


def test_published_release_drives_opening_and_on_request_images(tmp_path):
    task = next(task for task in get_tasks("base") if task.id == "retail_vision_031")
    raw_task = task.model_dump(mode="json", exclude_none=True)
    payloads = {
        "img_open_a": b"opening-a",
        "img_open_b": b"opening-b",
        "img_close": b"close-up",
    }
    release = tmp_path / "release"
    assets_dir = release / "assets"
    assets_dir.mkdir(parents=True)
    registry = {"schema_version": "tauvision-image-registry-v1", "assets": {}}
    for asset_id, payload in payloads.items():
        digest = hashlib.sha256(payload).hexdigest()
        (assets_dir / f"{digest}.png").write_bytes(payload)
        registry["assets"][asset_id] = {
            "path": f"assets/{digest}.png",
            "sha256": digest,
            "alt_text": "User-submitted product evidence photo.",
        }
    raw_task["image_bank"] = [
        {"kappa_index": 0, "asset_ids": ["img_open_a", "img_open_b"]},
        {"kappa_index": 1, "asset_ids": ["img_close"]},
    ]
    raw_task["image_triggers"][0]["asset_ids"] = ["img_open_a", "img_open_b"]
    raw_task["image_triggers"][1]["asset_ids"] = ["img_close"]
    (release / "tasks.json").write_text(json.dumps([raw_task]), encoding="utf-8")
    (release / "image_registry.json").write_text(
        json.dumps(registry), encoding="utf-8"
    )

    released = get_released_tasks(
        release,
        task_split_name=None,
        task_ids=["retail_vision_031"],
    )[0]
    assert "runtime_image_assets" not in released.model_dump(mode="json")
    fired: set[int] = set()
    opening = resolve_image_trigger(
        task=released,
        event="opening",
        agent_text="Hi! How can I help?",
        image_seed=1,
        fired_trigger_indices=fired,
    )
    assert opening is not None
    assert opening.asset_id == "img_open_b"
    fired.add(opening.trigger_index)

    assert (
        resolve_image_trigger(
            task=released,
            event="on_request",
            agent_text="Can you provide the serial number?",
            image_seed=1,
            fired_trigger_indices=fired,
        )
        is None
    )
    close_up = resolve_image_trigger(
        task=released,
        event="on_request",
        agent_text=(
            "Please send a clearer photo showing the burn damage and serial number."
        ),
        image_seed=1,
        fired_trigger_indices=fired,
    )
    assert close_up is not None
    assert close_up.asset_id == "img_close"

    message = UserMessage(
        role="user",
        content="Here is the requested photo.",
        image_content=close_up.image_content,
        image_alt=close_up.alt_text,
    )
    converted = to_litellm_messages([message])
    assert converted[0]["content"][0]["type"] == "text"
    assert converted[0]["content"][1]["type"] == "image_url"
    assert converted[0]["content"][1]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )
    simulator_view = UserState(
        system_messages=[SystemMessage(role="system", content="simulate user")],
        messages=[message],
    ).flip_roles()
    assert simulator_view[0].image_content is None


def test_published_release_rejects_changed_asset(tmp_path):
    task = get_tasks("base")[0].model_dump(mode="json", exclude_none=True)
    payload = b"original"
    digest = hashlib.sha256(payload).hexdigest()
    release = tmp_path / "release"
    assets_dir = release / "assets"
    assets_dir.mkdir(parents=True)
    asset_path = assets_dir / f"{digest}.png"
    asset_path.write_bytes(b"tampered")
    task["image_bank"] = [{"kappa_index": 0, "asset_ids": ["img_one"]}]
    task["image_triggers"][0]["asset_ids"] = ["img_one"]
    (release / "tasks.json").write_text(json.dumps([task]), encoding="utf-8")
    (release / "image_registry.json").write_text(
        json.dumps(
            {
                "schema_version": "tauvision-image-registry-v1",
                "assets": {
                    "img_one": {
                        "path": f"assets/{digest}.png",
                        "sha256": digest,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        get_released_tasks(release, task_split_name=None)
