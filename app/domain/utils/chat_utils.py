from langchain_core.messages import HumanMessage, AIMessage
from typing import List
from operator import gt, lt, eq, ge, le, ne

from app.api.exceptions import APIError


def flatten_dict(data, indent=0) -> str:
    lines = []
    if isinstance(data, dict):
        for k, v in data.items():
            key = k.replace("_", " ").title()
            if isinstance(v, dict):
                lines.append(" " * indent + f"{key}:")
                lines.append(flatten_dict(v, indent + 2))
            elif isinstance(v, str) and "\n" in v:  # multiline text
                lines.append(" " * indent + f"{key}:\n{' ' * (indent+2)}{v}")
            else:
                lines.append(" " * indent + f"{key}: {v}")
    else:
        lines.append(" " * indent + str(data))
    return "\n".join(lines)


def get_last_n_chats(messages, n=5):
    last_messages = messages[-n:]

    chat_history = []
    for msg in last_messages:
        if isinstance(msg, HumanMessage):
            chat_history.append({"role": "human", "content": msg.content})
        elif isinstance(msg, AIMessage):
            chat_history.append({"role": "ai", "content": msg.content})
        else:
            chat_history.append({"role": "other", "content": msg.content})

    return chat_history


def get_all_chats(messages):
    """Return the full chat history from a list of messages in role/content dict format."""
    chat_history = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            chat_history.append({"role": "human", "content": msg.content})
        elif isinstance(msg, AIMessage):
            chat_history.append({"role": "ai", "content": msg.content})
        else:
            chat_history.append({"role": "other", "content": msg.content})
    return chat_history


def compare_message_role_count(
    messages: List, role: str, op: str, threshold: int
) -> bool:
    """
    Compare count of messages for a given role using a comparison operator.
    :param messages: List of messages.
    :param role: Target role ('human', 'ai', 'other').
    :param op: Operator as string ('>', '<', '==', '>=', '<=', '!=').
    :param threshold: Number to compare against.
    :return: Boolean result of the comparison.
    """
    role_map = {
        "human": HumanMessage,
        "ai": AIMessage,
        "other": None,  # Fallback for unmatched types
    }
    if role not in role_map:
        raise APIError("Invalid role specified", status_code=400)
    if op not in {">", "<", "==", ">=", "<=", "!="}:
        raise APIError("Invalid comparison operator", status_code=400)

    # Select class or 'other'
    if role == "other":
        count = sum(
            1 for msg in messages if not isinstance(msg, (HumanMessage, AIMessage))
        )
    else:
        cls = role_map[role]
        count = sum(1 for msg in messages if isinstance(msg, cls))

    ops = {">": gt, "<": lt, "==": eq, ">=": ge, "<=": le, "!=": ne}
    return ops[op](count, threshold)
