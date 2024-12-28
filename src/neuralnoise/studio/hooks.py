import json
import logging
import os
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable

from autogen.agentchat import Agent

logger = logging.getLogger(__name__)


def save_last_json_message_hook(filename: str, output_dir: str | Path):
    os.makedirs(output_dir, exist_ok=True)

    def hook(sender, message, recipient, silent):
        message_dict = json.loads(message)
        message = json.dumps(message_dict, indent=2, ensure_ascii=False)

        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filepath = Path(output_dir) / f"{filename}_{date_str}.json"
        filepath.write_text(message)
        logger.debug(f"Saved agent message to {filepath}")

        return message

    return hook


def save_last_message_hook(filename: str, output_dir: str | Path):
    os.makedirs(output_dir, exist_ok=True)

    def hook(sender, message, recipient, silent):
        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filepath = Path(output_dir) / f"{filename}_{date_str}.txt"
        filepath.write_text(message)
        logger.debug(f"Saved agent message to {filepath}")

        return message

    return hook


Message = dict[str, Any]
Messages = list[Message]


def optimize_chat_history_hook(
    agents: list[Agent | str],
) -> Callable[[Messages], Messages]:
    """Optimizes the chat history by keeping the last message for each of the
    agents in the list.
    """

    @wraps(optimize_chat_history_hook)
    def hook(messages: Messages) -> Messages:
        last_messages = {}
        agents_names = {
            agent.name if isinstance(agent, Agent) else agent for agent in agents
        }

        new_messages: list[dict[str, Any]] = []
        for message in reversed(messages):
            agent_name = message["name"]

            if agent_name in agents_names:
                if agent_name not in last_messages:
                    last_messages[agent_name] = message
                    new_messages.append(message)
            else:
                new_messages.append(message)

        logger.debug(
            f"On optimize_chat_history hook, #messages: {len(messages)} "
            f"-> #optimized-messages: {len(new_messages)}"
        )

        return list(reversed(new_messages))

    return hook
