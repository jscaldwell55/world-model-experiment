# models/tools.py
from pydantic import BaseModel, Field
from typing import Literal, Optional, Any


class ToolCall(BaseModel):
    """Structured tool call for parsing LLM outputs"""
    tool_name: str = Field(description="Name of the tool to call")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Tool parameters")


class ToolResult(BaseModel):
    """Structured tool result"""
    success: bool = Field(description="Whether tool execution succeeded")
    observation: dict = Field(description="Observation returned from tool")
    error: Optional[str] = Field(default=None, description="Error message if failed")


# Environment-specific tool classes
class HotPotTools:
    """Tools for HotPotLab environment"""

    @staticmethod
    def measure_temp() -> dict:
        """Measure the pot temperature with a thermometer"""
        return {'tool': 'measure_temp', 'params': {}}

    @staticmethod
    def wait(seconds: int = 10) -> dict:
        """Wait for specified seconds, allowing temperature to evolve"""
        return {'tool': 'wait', 'params': {'seconds': seconds}}

    @staticmethod
    def touch_pot() -> dict:
        """Touch the pot (careful - might burn if hot!)"""
        return {'tool': 'touch_pot', 'params': {}}

    @staticmethod
    def toggle_stove() -> dict:
        """Cycle stove power: off -> low -> high -> off"""
        return {'tool': 'toggle_stove', 'params': {}}

    @staticmethod
    def get_tool_descriptions() -> str:
        """Return formatted tool descriptions for prompts"""
        return """Available Tools:
- measure_temp(): Measure pot temperature with thermometer (noisy reading)
- wait(seconds): Wait for time to pass, temperature evolves
- touch_pot(): Touch the pot to feel temperature (burns if > 60C)
- toggle_stove(): Cycle stove power (off -> low -> high -> off)"""


class SwitchLightTools:
    """Tools for SwitchLight environment"""

    @staticmethod
    def flip_switch() -> dict:
        """Flip the switch and observe the light"""
        return {'tool': 'flip_switch', 'params': {}}

    @staticmethod
    def jiggle_relay() -> dict:
        """Jiggle the relay (might fix faulty connection)"""
        return {'tool': 'jiggle_relay', 'params': {}}

    @staticmethod
    def inspect_wires() -> dict:
        """Inspect wiring (costs -1 reward, gives hints)"""
        return {'tool': 'inspect_wires', 'params': {}}

    @staticmethod
    def observe_light() -> dict:
        """Check if light is currently on"""
        return {'tool': 'observe_light', 'params': {}}

    @staticmethod
    def get_tool_descriptions() -> str:
        """Return formatted tool descriptions for prompts"""
        return """Available Tools:
- flip_switch(): Toggle switch position and observe light
- jiggle_relay(): Attempt to fix potentially faulty relay
- inspect_wires(): Inspect wiring (costly but provides hints)
- observe_light(): Check current light state"""


class ChemTileTools:
    """Tools for ChemTile environment"""

    @staticmethod
    def mix(compound_a: str, compound_b: str) -> dict:
        """Mix two compounds to attempt a reaction"""
        return {
            'tool': 'mix',
            'params': {'compound_a': compound_a, 'compound_b': compound_b}
        }

    @staticmethod
    def heat() -> dict:
        """Increase temperature (increases reaction speed and explosion risk)"""
        return {'tool': 'heat', 'params': {}}

    @staticmethod
    def cool() -> dict:
        """Decrease temperature (reduces reaction speed and explosion risk)"""
        return {'tool': 'cool', 'params': {}}

    @staticmethod
    def inspect(compound: str) -> dict:
        """Get information about a specific compound"""
        return {'tool': 'inspect', 'params': {'compound': compound}}

    @staticmethod
    def get_tool_descriptions() -> str:
        """Return formatted tool descriptions for prompts"""
        return """Available Tools:
- mix(compound_a, compound_b): Mix two compounds (may produce new compound or explode)
- heat(): Increase temperature (faster reactions, more explosions)
- cool(): Decrease temperature (slower reactions, fewer explosions)
- inspect(compound): Get information about a compound's properties"""


# Tool registry mapping environment names to tool classes
TOOL_REGISTRY = {
    'HotPotLab': HotPotTools,
    'SwitchLight': SwitchLightTools,
    'ChemTile': ChemTileTools,
}


def get_tools_for_environment(env_name: str) -> Any:
    """
    Get tool class for environment.

    Args:
        env_name: Environment class name

    Returns:
        Tool class with static methods

    Raises:
        ValueError: If environment not found
    """
    if env_name not in TOOL_REGISTRY:
        raise ValueError(f"No tools registered for environment: {env_name}")

    return TOOL_REGISTRY[env_name]


def parse_tool_call(action_string: str) -> Optional[ToolCall]:
    """
    Parse action string like "measure_temp()" or "wait(10)" into ToolCall.

    Args:
        action_string: String like "tool_name(param1, param2)"

    Returns:
        ToolCall object or None if parsing fails
    """
    if not action_string:
        return None

    import re

    # Match tool_name() or tool_name(args)
    match = re.match(r'(\w+)\((.*?)\)', action_string.strip())

    if not match:
        return None

    tool_name = match.group(1)
    args_str = match.group(2).strip()

    # Parse arguments
    parameters = {}
    if args_str:
        # Simple parsing: split by comma, handle basic types
        args = [arg.strip().strip('"').strip("'") for arg in args_str.split(',')]

        # Try to infer parameter names from common patterns
        if tool_name == 'wait' and len(args) == 1:
            try:
                parameters['seconds'] = int(args[0])
            except ValueError:
                parameters['seconds'] = 10  # Default

        elif tool_name == 'mix' and len(args) == 2:
            parameters['compound_a'] = args[0]
            parameters['compound_b'] = args[1]

        elif tool_name == 'inspect' and len(args) == 1:
            parameters['compound'] = args[0]

    return ToolCall(tool_name=tool_name, parameters=parameters)


def execute_tool_call(tool_call: ToolCall, environment) -> ToolResult:
    """
    Execute a tool call on an environment.

    Args:
        tool_call: Parsed tool call
        environment: Environment instance

    Returns:
        ToolResult with observation or error
    """
    try:
        # Construct action string for environment
        if tool_call.parameters:
            # Format parameters
            if tool_call.tool_name == 'wait':
                action_str = f"wait({tool_call.parameters.get('seconds', 10)})"
            elif tool_call.tool_name == 'mix':
                compound_a = tool_call.parameters.get('compound_a', 'A')
                compound_b = tool_call.parameters.get('compound_b', 'B')
                action_str = f"mix({compound_a}, {compound_b})"
            elif tool_call.tool_name == 'inspect':
                compound = tool_call.parameters.get('compound', 'A')
                action_str = f"inspect({compound})"
            else:
                action_str = f"{tool_call.tool_name}()"
        else:
            action_str = tool_call.tool_name

        # Execute on environment
        observation, reward, done, info = environment.step(action_str)

        return ToolResult(
            success=True,
            observation=observation
        )

    except Exception as e:
        return ToolResult(
            success=False,
            observation={},
            error=str(e)
        )
