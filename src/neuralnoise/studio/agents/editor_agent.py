from autogen import AssistantAgent


def create_editor_agent(
    system_msg: str,
    llm_config: dict,
) -> AssistantAgent:
    """Create and return an EditorAgent that reviews a section script and determines the next agent.

    Args:
        system_msg (str): The system message for the EditorAgent.
        llm_config (dict): The LLM configuration.

    Returns:
        AssistantAgent: The EditorAgent instance.
    """

    agent = AssistantAgent(
        name="EditorAgent",
        system_message=system_msg,
        llm_config=llm_config,
    )

    return agent
