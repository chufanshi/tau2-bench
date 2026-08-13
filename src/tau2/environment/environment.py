import json
from copy import deepcopy
from datetime import date, datetime
from typing import Any, Literal, Optional

from loguru import logger
from pydantic import BaseModel, Field

from tau2.data_model.message import (
    AssistantMessage,
    Message,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.data_model.tasks import EnvAssertion, EnvFunctionCall, InitializationData
from tau2.environment.db import DB
from tau2.environment.tool import Tool
from tau2.environment.toolkit import ToolKitBase, ToolSignature, get_tool_signatures


# tau-vision conflict arm: tools that report the same fact as the status bar,
# in prose rather than through a structured kappa. Under TAU2_STRICT_READOUT
# the simulated user narrates these too, so a lie confined to the status bar
# would be contradicted by the user's own next sentence — they would appear to
# misread one screen and read the next one correctly, which is not the persona
# the design calls for ("the user consistently misreads this one fact").
# Each entry maps the disputed field to the readouts that must move with it.
# Ordered (pattern, replacement) rules, first match wins — the "fault present"
# shapes are listed before the "fault absent" one so a rewritten sentence is
# never re-matched and flipped back. Patterns must consume the whole clause
# they replace: check_wifi_status and check_vpn_status embed an SSID / detail
# string in their positive forms, and a prefix-only replacement would leave
# that fragment stranded after the negated sentence.
_CONFLICT_TEXT_FLIPS: dict[str, dict[str, list[tuple[str, str]]]] = {
    "data_saver_mode": {
        "check_data_restriction_status": [
            (r"Data Saver mode is ON \(limits data usage\)\.", "Data Saver mode is OFF."),
            (r"Data Saver mode is OFF\.", "Data Saver mode is ON (limits data usage)."),
        ],
    },
    "wifi_enabled": {
        "check_wifi_status": [
            (r"Wi-Fi is ON and connected to .*?\. Signal strength: [^.]*\.",
             "Wi-Fi is turned OFF."),
            (r"Wi-Fi is ON but not connected to any network\.", "Wi-Fi is turned OFF."),
            (r"Wi-Fi is turned OFF\.", "Wi-Fi is ON but not connected to any network."),
        ],
    },
    # The disputed field is vpn_CONNECTED, not the VPN toggle. Negating a
    # connected VPN therefore yields "on in settings but not connected" rather
    # than "turned OFF" — flipping the toggle instead would misstate a second
    # fact, and the design allows exactly one.
    "vpn_connected": {
        "check_vpn_status": [
            (r"VPN is ON and connected\. Details: .*",
             "VPN is turned ON in settings, but currently not connected."),
            (r"VPN is ON and connected \(no specific details available\)\.",
             "VPN is turned ON in settings, but currently not connected."),
            (r"VPN is turned ON in settings, but currently not connected\.",
             "VPN is ON and connected (no specific details available)."),
            (r"VPN is turned OFF\.",
             "VPN is ON and connected (no specific details available)."),
        ],
    },
}


def _conflict_flip_text(field: str, tool: str, text: str) -> str:
    """Flip the disputed fact inside a prose readout, keeping the rest intact.

    Returns the text unchanged when this tool does not report the disputed
    field, so it is safe to call on every readout.
    """
    import re as _re

    for pattern, replacement in _CONFLICT_TEXT_FLIPS.get(field, {}).get(tool, []):
        new = _re.sub(pattern, replacement, text, count=1)
        if new != text:
            return new
    return text


def _conflict_readout(kappa: dict, field: str, lie: bool) -> str:
    """tau-vision conflict arm: textual screen readout from a tool's kappa.

    With lie=True the disputed boolean field is flipped — the simulated user
    'misread' exactly one fact on their screen. Everything else is truthful.
    """
    k = dict(kappa)
    if lie and field in k and isinstance(k[field], bool):
        k[field] = not k[field]

    def fmt(v: Any) -> str:
        if isinstance(v, bool):
            return "On" if v else "Off"
        return str(v)

    lines = [
        f"{key.replace('_', ' ').title()}: {fmt(val)}"
        for key, val in k.items()
        if val is not None and key != "source_text"
    ]
    return "You read the following on your screen:\n" + "\n".join(lines)


class EnvironmentInfo(BaseModel):
    """
    Environment information.
    """

    domain_name: str = Field(description="The name of the domain.")
    policy: str = Field(description="The policy of the agent.")
    tool_defs: Optional[dict[str, ToolSignature]] = Field(
        description="The tool definitions of the environment.", default=None
    )


class Environment:
    """
    Environment
    """

    def __init__(
        self,
        domain_name: str,
        policy: str,
        tools: Optional[ToolKitBase] = None,
        user_tools: Optional[ToolKitBase] = None,
        solo_mode: bool = False,
    ):
        """
        Environment
        Args:
            domain_name: The name of the domain.
            policy: The policy of the domain.
            tools: The tools available to the assistant in the domain.
            user_tools: The tools available to the user in the domain.
            solo_mode: The agent will have access to both user and assistant tools.
        """
        self.domain_name = domain_name
        self.policy = policy
        self.tools = tools
        self.user_tools = user_tools
        self.solo_mode = solo_mode
        if self.solo_mode:
            self.validate_solo_mode()
        self.sync_tools()

    def get_domain_name(self) -> str:
        """
        Get the name of the domain.
        """
        return self.domain_name

    def get_policy(self) -> str:
        """
        Get the policy of the domain.
        """
        return self.policy

    def get_tools(self) -> list[Tool]:
        """
        Get the tools of the domain.
        """
        if self.tools is None:
            raise ValueError("Tools not available")
        return list(self.tools.get_tools().values())

    def get_user_tools(self, include: Optional[list[str]] = None) -> list[Tool]:
        """
        Get the user tools of the domain, optionally filtered by name.

        Args:
            include: If provided, only return tools whose names are in this list.
                If None, return all user tools (no filtering).

        Returns:
            A list of Tool objects available to the user.
        """
        if self.user_tools is None:
            raise ValueError("User tools not available")
        return list(self.user_tools.get_tools(include=include).values())

    def get_tools_description(
        self, env_type: Literal["user", "assistant"]
    ) -> Optional[str]:
        """
        Return a description of the user tools.
        """
        if env_type == "user":
            tool_kit = self.user_tools
        elif env_type == "assistant":
            tool_kit = self.tools
        else:
            raise ValueError(f"Invalid environment type: {env_type}")
        if tool_kit is None:
            return None
        tools = sorted(tool_kit.get_tools().values(), key=lambda x: x.name)
        return "\n\n".join(
            [f"{i + 1}. {t.name}\n{t.short_desc}" for i, t in enumerate(tools)]
        )

    def _has_tool(self, tool_name: str) -> bool:
        """Check if a tool exists in the environment.

        Checks toolkit tools and user tools.
        """
        if self.tools is not None and self.tools.has_tool(tool_name):
            return True
        if self.user_tools is not None and self.user_tools.has_tool(tool_name):
            return True
        return False

    def _is_mutating_tool(self, tool_name: str) -> bool:
        """Check if a tool mutates environment state.

        Looks up ``mutates_state`` on the underlying function via the toolkit.
        Falls back to ``True`` (assume mutation) if the tool or attribute
        cannot be found.
        """
        for toolkit in (self.tools, self.user_tools):
            if toolkit is not None and toolkit.has_tool(tool_name):
                return toolkit.tool_mutates_state(tool_name)
        return True  # safe fallback: assume mutation

    def use_tool(self, tool_name: str, **kwargs) -> Any:
        """
        Use a tool available to the assistant of the domain.
        """
        if self.tools is None:
            raise ValueError("Tools not available")
        return self.tools.use_tool(tool_name=tool_name, **kwargs)

    def use_user_tool(self, tool_name: str, **kwargs) -> Any:
        """
        Use a tool available to the user of the domain.
        """
        if self.user_tools is None:
            raise ValueError("User tools not available")
        return self.user_tools.use_tool(tool_name=tool_name, **kwargs)

    def make_tool_call(
        self,
        tool_name: str,
        requestor: Literal["user", "assistant"] = "assistant",
        **kwargs,
    ) -> Any:
        """
        Make a tool call based on the requestor.
        Args:
            tool_name: The name of the tool to call.
            requestor: The requestor of the tool call.
            kwargs: The arguments to pass to the tool.
        Returns:
            The response of the tool call.

        Note: This does not call sync_tools.
        """
        if requestor == "user":
            if self.solo_mode:
                raise ValueError("User tool calls are not allowed in solo mode")
            return self.use_user_tool(tool_name=tool_name, **kwargs)
        elif requestor == "assistant":
            if self.solo_mode and self.user_tools is not None:
                if self.user_tools.has_tool(tool_name):
                    return self.use_user_tool(tool_name=tool_name, **kwargs)
            return self.use_tool(tool_name=tool_name, **kwargs)
        else:
            raise ValueError(f"Invalid requestor: {requestor}")

    def sync_tools(self):
        """
        Sync the user and assistant tools.
        Subclass should override this method if tools need to be synced.
        """
        pass

    def run_env_function_call(self, env_function_call: EnvFunctionCall) -> Any:
        """
        Runs any function available on agent environment or user environment.
        """
        env_type = env_function_call.env_type
        func_name = env_function_call.func_name
        if env_type == "user":
            tool_kit = self.user_tools
        elif env_type == "assistant":
            tool_kit = self.tools
        else:
            raise ValueError(f"Invalid environment type: {env_type}")
        func = getattr(tool_kit, func_name)
        if func is None:
            raise ValueError(f"Function {func_name} not found in {env_type} tools")
        res = func(**env_function_call.arguments)
        self.sync_tools()
        return res

    def run_env_assertion(
        self,
        assertion: EnvAssertion,
        raise_assertion_error: bool = True,
    ) -> bool:
        """
        Runs any assertion function on agent tools or user tools.
        """
        if not isinstance(assertion, EnvAssertion):
            raise ValueError(f"Assertion must be an EnvAssertion. Got {assertion}")
        res = self.run_env_function_call(assertion)
        if not isinstance(res, bool):
            raise ValueError(
                f"Function {assertion.func_name} returned {type(res)} instead of bool"
            )
        assert_pass = res == assertion.assert_value
        if raise_assertion_error:
            assert assert_pass, assertion.message or f"Assertion failed: {assertion}"
        return assert_pass

    def run_env_function_calls(self, env_function_calls: list[EnvFunctionCall]) -> None:
        """
        Run a list of environment function calls. If the function call is an assertion,
        an assertion check will be performed.
        """
        for env_function_call in env_function_calls:
            if isinstance(env_function_call, EnvAssertion):
                self.run_env_assertion(env_function_call, raise_assertion_error=True)
            else:
                self.run_env_function_call(env_function_call)

    def get_info(self, include_tool_info: bool = False) -> EnvironmentInfo:
        """
        Get environment information.
        """
        return EnvironmentInfo(
            domain_name=self.domain_name,
            policy=self.policy,
            tool_defs=(
                get_tool_signatures(self.tools)
                if self.tools is not None and include_tool_info
                else None
            ),
            user_tool_defs=(
                get_tool_signatures(self.user_tools)
                if self.user_tools is not None and include_tool_info
                else None
            ),
        )

    def check_db(self, reference: DB) -> bool:
        """
        Compare the agent database with the reference
        """
        return self.get_db_hash() == reference.get_hash()

    def check_user_db(self, reference: DB) -> bool:
        """
        Compare the user database with the reference
        """
        return self.get_user_db_hash() == reference.get_hash()

    def get_db_hash(self) -> Optional[str]:
        """
        Get a hash of the agent database
        Returns None if the database is not available
        """
        if self.tools is None:
            return None
        return self.tools.get_db_hash()

    def get_user_db_hash(self) -> Optional[str]:
        """
        Get a hash of the user database
        Returns None if the database is not available
        """
        if self.user_tools is None:
            return None
        return self.user_tools.get_db_hash()

    def set_state(
        self,
        initialization_data: Optional[InitializationData],
        initialization_actions: Optional[list[EnvFunctionCall]],
        message_history: list[Message],
        strict: bool = True,
    ):
        """
        Set the state of the environment given initialization data and a list of messages.

        Args:
            strict: When True (default), raise if a replayed mutating tool call
                returns different content than the recorded ToolMessage. When
                False, log a warning instead and continue the replay. Lenient
                mode is intended for re-grading historical trajectories whose
                recorded tool outputs contain cosmetic drift against current
                tool code (e.g. numeric argument echoes rendered as `25` by the
                code that produced them but `25.0` after numeric-argument
                normalization); the state mutation is applied identically
                either way.
        """
        if self.solo_mode:
            assert all(
                [not isinstance(message, UserMessage) for message in message_history]
            ), "User messages are not allowed in solo mode"

        def get_actions_from_messages(
            messages: list[Message],
        ) -> list[tuple[ToolCall, ToolMessage]]:
            """
            Get the actions from the messages.
            """
            messages = deepcopy(messages)[::-1]
            actions = []
            while messages:
                message = messages.pop()
                if isinstance(message, ToolMessage):
                    raise ValueError(
                        "Tool message not expected. Tool messages should always follow a tool call."
                    )
                if (
                    isinstance(message, (AssistantMessage, UserMessage))
                    and message.is_tool_call()
                ):
                    tool_calls = message.tool_calls
                    for tc in tool_calls:
                        if len(messages) == 0:
                            raise ValueError("Tool message expected. Got None.")
                        tm = messages.pop()
                        if not isinstance(tm, ToolMessage):
                            raise ValueError(f"Tool message expected. Got {type(tm)}")
                        if tc.id != tm.id:
                            raise ValueError(
                                f"Tool call id mismatch. Got {tc.id} and {tm.id}"
                            )
                        actions.append((tc, tm))

            return actions

        if initialization_data is not None:
            if initialization_data.agent_data is not None:
                self.tools.update_db(initialization_data.agent_data)
                # Sync user_tools.db to point to the same db instance as tools.db
                # This is necessary because update_db creates a new db instance
                if self.user_tools is not None and self.user_tools.db is not None:
                    self.user_tools.db = self.tools.db
            if initialization_data.user_data is not None:
                self.user_tools.update_db(initialization_data.user_data)
                # Sync tools.db to point to the same db instance as user_tools.db
                if self.tools is not None and self.tools.db is not None:
                    self.tools.db = self.user_tools.db

        if initialization_actions is not None:
            for action in initialization_actions:
                self.run_env_function_call(action)

        action_responses = get_actions_from_messages(message_history)
        for tool_call, expected_response in action_responses:
            if not self._has_tool(tool_call.name):
                # Hallucinated tool name. The live env returned a
                # ToolMessage(error=True) for this call and made no state
                # change, so replay it as a no-op. The agent's subsequent
                # recovery (if any) will still be replayed and determine
                # the final state. Repeated hallucination is bounded
                # upstream by the orchestrator's max_errors guard, which
                # ends the live sim with TerminationReason.TOO_MANY_ERRORS
                # before evaluation runs.
                logger.debug(
                    f"Skipping unknown tool '{tool_call.name}' during replay "
                    "(no-op, matching live env behavior on hallucinated tools)."
                )
                continue
            # Non-mutating tools (reads, thinks, etc.) don't change state --
            # skip them to avoid re-execution and non-deterministic output
            # comparison issues.
            if not self._is_mutating_tool(tool_call.name):
                continue
            response = self.get_response(tool_call)
            try:
                content = json.loads(response.content)
            except json.JSONDecodeError:
                content = response.content
            try:
                expected_content = json.loads(expected_response.content)
            except json.JSONDecodeError:
                expected_content = expected_response.content
            if content != expected_content:
                if strict:
                    raise ValueError(
                        f"Tool call:\n{tool_call}\n\nReturned:\n{response}\n\nExpected:\n{expected_response}"
                    )
                logger.warning(
                    f"Replayed tool call '{tool_call.name}' returned different "
                    f"content than the recorded ToolMessage; continuing because "
                    f"strict=False. Recorded output may predate current tool "
                    f"code.\nTool call:\n{tool_call}"
                )
        self.sync_tools()

    @classmethod
    def to_json_str(cls, resp: Any) -> str:
        """
        Convert a response to a JSON string.
        """

        def _process(resp: Any) -> str:
            if isinstance(resp, BaseModel):
                return resp.model_dump()
            elif isinstance(resp, str):
                return resp
            elif resp is None:
                return resp
            elif isinstance(resp, (int, float, bool)):
                return str(resp)
            elif isinstance(resp, list):
                return [_process(item) for item in resp]
            elif isinstance(resp, tuple):
                return tuple(_process(item) for item in resp)
            elif isinstance(resp, dict):
                return {k: _process(v) for k, v in resp.items()}
            elif isinstance(resp, (datetime, date)):
                # TODO: this did not fix the error: Object of type date is not JSON serializable
                return resp.isoformat()
            else:
                raise ValueError(f"Unsupported type: {type(resp)}")

        if not isinstance(resp, str):
            return json.dumps(_process(resp), default=str)  # FIXME: add default=str
        return resp

    def set_solo_mode(self, solo_mode: bool):
        """
        Set the solo mode of the environment.
        """
        self.solo_mode = solo_mode
        if solo_mode:
            self.validate_solo_mode()

    def validate_solo_mode(self) -> None:
        """
        Validate the tool call in solo mode.
        """
        assistant_tool_names = set(self.tools.get_tools().keys())
        user_tool_names = (
            set(self.user_tools.get_tools().keys())
            if self.user_tools is not None
            else set()
        )
        overlap = assistant_tool_names & user_tool_names
        if len(overlap) > 0:
            raise ValueError(f"Tool names overlap: {overlap}")

    def get_response(self, message: ToolCall) -> ToolMessage:
        """
        Get the response of the domain. This also calls sync_tools.
        Args:
            message: The message to get the response for.
        Returns:
            The response of the tool call.
        """
        error = False
        try:
            resp = self.make_tool_call(
                message.name, requestor=message.requestor, **message.arguments
            )
            self.sync_tools()
        except Exception as e:
            resp = f"Error: {e}"
            error = True
        logger.debug(f"Response: {resp}")
        from tau2.data_model.image import ImageObservation

        # tau-vision STRICT monopoly mode: render listed user-side observation
        # tools' text output as a phone-panel screenshot (facts pixels-only).
        import os as _os

        if (
            _os.environ.get("TAU2_OBSERVATION_MODALITY") == "vision_strict"
            and message.requestor == "user"
            and isinstance(resp, str)
            and not error
            and message.name
            in set(
                _os.environ.get(
                    "TAU2_STRICT_VISUAL_TOOLS",
                    "run_speed_test,check_sim_status,check_data_restriction_status,"
                    "check_apn_settings,check_wifi_status,check_wifi_calling_status,"
                    "check_vpn_status,check_network_mode_preference,check_app_status,"
                    "check_app_permissions,can_send_mms",
                ).split(",")
            )
        ):
            import base64 as _b64
            import io as _io

            from tauvision.renderers.phone_panel import render_phone_panel

            img = render_phone_panel(
                message.name, resp, image_seed=int(_os.environ.get("TAU2_IMAGE_SEED", "0"))
            )
            buf = _io.BytesIO()
            img.save(buf, format="PNG")
            # TAU2_STRICT_READOUT=1: the simulated user narrates every screen
            # they forward, so the alt carries the real readout instead of a
            # neutral placeholder. Needed by the conflict experiment, where the
            # user's narration is the channel the lie travels on — with the
            # default placeholder the user has nothing to say about these
            # screens and the lie would be the only utterance in the run.
            # OFF by default: the main experiment's vision_strict arms are the
            # "no narration" condition and must stay byte-identical.
            _readout = resp
            if _os.environ.get("TAU2_STRICT_READOUT") == "1":
                # Keep the lie consistent across every screen that reports the
                # disputed fact (see _CONFLICT_TEXT_FLIPS). The screenshot
                # itself stays truthful — only the narration moves.
                if (
                    _os.environ.get("TAU2_CONFLICT") == "1"
                    and _os.environ.get("TAU2_CONFLICT_LIE", "1") == "1"
                ):
                    _readout = _conflict_flip_text(
                        _os.environ.get("TAU2_CONFLICT_FIELD", "airplane_mode"),
                        message.name,
                        _readout,
                    )
            else:
                _readout = (
                    f"Screenshot of the phone screen for {message.name} attached."
                )
            resp = ImageObservation(
                image_b64=_b64.b64encode(buf.getvalue()).decode(),
                alt_text=_readout,
                kappa={"source_text": resp},
            )

        # tau-vision Line 3 (banking-vision): retrieval payloads as document
        # pages. The SEARCH ran over the text KB exactly as in the text arm;
        # only the returned payload is rendered to pixels. Gated by
        # TAU2_KB_VISION=1; tool list via TAU2_KB_VISUAL_TOOLS.
        if (
            _os.environ.get("TAU2_KB_VISION") == "1"
            and message.requestor == "assistant"
            and isinstance(resp, str)
            and not error
            and message.name
            in set(_os.environ.get("TAU2_KB_VISUAL_TOOLS", "KB_search,grep").split(","))
        ):
            import base64 as _b64
            import io as _io

            from tauvision.renderers.document_page import render_document_pages

            _imgs = render_document_pages(
                resp,
                image_seed=int(_os.environ.get("TAU2_IMAGE_SEED", "0")),
                title=f"Knowledge Base — {message.name}",
            )
            _pages = []
            for _im in _imgs:
                _buf = _io.BytesIO()
                _im.save(_buf, format="PNG")
                _pages.append(_b64.b64encode(_buf.getvalue()).decode())
            resp = ImageObservation(
                image_b64=_pages[0],
                image_pages=_pages,
                alt_text=f"{len(_pages)} knowledge-base document page(s) attached.",
                kappa={"source_text": resp},
            )

        # tau-vision CONFLICT arm: the textual readout (the simulator's only
        # information channel) is derived from the tool's kappa with one
        # disputed field flipped; the screenshot stays truthful and rides on
        # to the agent. TAU2_CONFLICT_LIE=0 gives the truthful-readout control
        # arm over the identical pipeline.
        if (
            _os.environ.get("TAU2_CONFLICT") == "1"
            and message.requestor == "user"
            and isinstance(resp, ImageObservation)
            and message.name
            in set(
                _os.environ.get(
                    "TAU2_CONFLICT_TOOLS", "check_status_bar,check_network_status"
                ).split(",")
            )
            and isinstance(getattr(resp, "kappa", None), dict)
        ):
            resp.alt_text = _conflict_readout(
                resp.kappa,
                field=_os.environ.get("TAU2_CONFLICT_FIELD", "airplane_mode"),
                lie=_os.environ.get("TAU2_CONFLICT_LIE", "1") == "1",
            )

        if isinstance(resp, ImageObservation):
            return ToolMessage(
                id=message.id,
                content=resp.alt_text,
                image_content=resp.image_b64,
                image_alt=resp.alt_text,
                image_pages=getattr(resp, "image_pages", None),
                requestor=message.requestor,
                role="tool",
                error=error,
            )
        resp = self.to_json_str(resp)
        return ToolMessage(
            id=message.id,
            content=resp,
            requestor=message.requestor,
            role="tool",
            error=error,
        )
