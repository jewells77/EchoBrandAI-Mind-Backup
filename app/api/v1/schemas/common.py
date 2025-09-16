from langchain_core.messages import HumanMessage, AIMessage


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
