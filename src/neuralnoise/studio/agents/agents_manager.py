"""
AgentsManager module: Centralizes instantiation and handoff registration for swarm agents.
It also defines a final workflow to run the swarm chat and extract the final results.
"""

from pathlib import Path
from typing import Sequence, Tuple

from autogen import (
    AfterWork,
    AfterWorkOption,
    AssistantAgent,
    ChatResult,
    ConversableAgent,
    OnCondition,
    UserProxyAgent,
    initiate_swarm_chat,
    register_hand_off,
)

from neuralnoise.models import ContentAnalysis, PodcastScript
from neuralnoise.studio.agents.content_analyzer_agent import (
    create_content_analyzer_agent,
)
from neuralnoise.studio.agents.context_manager import SharedContext
from neuralnoise.studio.agents.editor_agent import create_editor_agent
from neuralnoise.studio.agents.planner_agent import create_planner_agent
from neuralnoise.studio.agents.script_generator_agent import (
    create_script_generator_agent,
)


class AgentsManager:
    def __init__(
        self,
        system_msgs: dict[str, str],
        llm_config: dict,
        language: str,
        work_dir: Path,
    ) -> None:
        """
        Initialize the AgentsManager with required configuration parameters,
        instantiate all agents, and register handoffs.

        Args:
            system_msgs (dict[str, str]): Dictionary with system messages for each agent.
            llm_config (dict): LLM configuration parameters.
            language (str): Language identifier for prompts.
        """
        self.work_dir = work_dir
        self.language: str = language
        self.llm_config: dict = llm_config
        self.agents: dict[str, AssistantAgent] = {}

        # Create a specific LLM config for ContentAnalyzerAgent with structured output
        content_analyzer_llm_config = llm_config.copy()
        content_analyzer_llm_config["response_format"] = ContentAnalysis

        # Create a specific LLM config for ScriptGeneratorAgent with structured output
        script_generator_llm_config = llm_config.copy()
        script_generator_llm_config["response_format"] = PodcastScript

        # Instantiate agents with placeholder next_agent dependencies.
        self.agents["PlannerAgent"] = create_planner_agent(
            system_msg=system_msgs.get("PlannerAgent", ""),
            llm_config=llm_config,
        )

        self.agents["ContentAnalyzerAgent"] = create_content_analyzer_agent(
            system_msg=system_msgs.get("ContentAnalyzerAgent", ""),
            llm_config=content_analyzer_llm_config,  # Use the modified config with response_format
            language=language,
            next_agent=self.agents["PlannerAgent"],
        )

        self.agents["ScriptGeneratorAgent"] = create_script_generator_agent(
            system_msg=system_msgs.get("ScriptGeneratorAgent", ""),
            llm_config=script_generator_llm_config,
            work_dir=self.work_dir,
            next_agent="EditorAgent",
        )

        self.agents["EditorAgent"] = create_editor_agent(
            system_msg=system_msgs.get("EditorAgent", ""),
            llm_config=llm_config,
        )

        # After all agents are created, register handoffs.
        self.register_handoffs()

    def register_handoffs(self) -> None:
        """
        Register handoffs between agents using OnCondition and AfterWork.
        This defines the control flow between agents during the swarm chat.
        """
        # ContentAnalyzerAgent -> PlannerAgent
        register_hand_off(
            self.agents["ContentAnalyzerAgent"],
            [
                OnCondition(
                    self.agents["PlannerAgent"],
                    "After content analysis, transfer to planning",
                )
            ],
        )
        # PlannerAgent -> ScriptGeneratorAgent
        register_hand_off(
            self.agents["PlannerAgent"],
            [
                OnCondition(
                    self.agents["ScriptGeneratorAgent"],
                    "Always transfer to ScriptGeneratorAgent",
                ),
                AfterWork(agent=self.agents["ScriptGeneratorAgent"]),
            ],
        )
        # ScriptGeneratorAgent -> EditorAgent
        register_hand_off(
            self.agents["ScriptGeneratorAgent"],
            [
                OnCondition(
                    self.agents["EditorAgent"],
                    "After writing a podcast section, always transfer to editor for review of the generated content",
                ),
                AfterWork(agent=self.agents["EditorAgent"]),
            ],
        )
        # EditorAgent: if revisions needed, transfer back to ScriptGeneratorAgent; otherwise, if more sections exist, transfer to PlannerAgent.
        register_hand_off(
            self.agents["EditorAgent"],
            [
                OnCondition(
                    self.agents["ScriptGeneratorAgent"],
                    "If changes are required, transfer back to Script Generator",
                ),
                AfterWork(agent=self.agents["PlannerAgent"]),
            ],
        )

    def run_swarm_chat(
        self, initial_message: str
    ) -> Tuple[ChatResult, SharedContext, ConversableAgent]:
        """
        Set up the shared state and user proxy, and initiate the swarm chat flow.

        Args:
            initial_message (str): The initial message or prompt for the swarm chat.

        Returns:
            Tuple[ChatResult, SharedState, ConversableAgent]: A tuple containing the conversation log,
            the final shared state, and the last agent that handled the conversation.
        """
        # Instantiate shared state.
        shared_state = SharedContext()

        # Initialize content from initial message if not already set
        if not shared_state.content and initial_message:
            shared_state.update_content(initial_message)

        # Prepare list of agents.
        swarm_agents: Sequence[ConversableAgent] = list(self.agents.values())

        # Initiate swarm chat starting with the ContentAnalyzerAgent.
        chat_result, final_context, last_agent = initiate_swarm_chat(
            initial_agent=self.agents["ContentAnalyzerAgent"],
            agents=swarm_agents,
            messages=initial_message,
            context_variables=shared_state.model_dump(),
            after_work=AfterWorkOption.SWARM_MANAGER,
            max_rounds=100,
        )

        # Update shared state with final context.
        shared_state = shared_state.model_validate(final_context)

        return chat_result, shared_state, last_agent
