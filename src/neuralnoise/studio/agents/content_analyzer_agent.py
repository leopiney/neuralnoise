from typing import Any

from autogen import AssistantAgent, SwarmResult

from neuralnoise.models import ContentAnalysis
from neuralnoise.studio.agents.context_manager import SharedContext


def create_content_analyzer_agent(
    system_msg: str,
    llm_config: dict,
    language: str,
    next_agent: AssistantAgent | None,
) -> AssistantAgent:
    """Create and return a ContentAnalyzerAgent for analyzing content."""

    def save_original_content(content: str, context_variables: dict) -> SwarmResult:
        """This function saves the original content to the shared state."""
        shared_state = SharedContext.model_validate(context_variables)
        shared_state.update_content(content)

        return SwarmResult(
            values=content,
            agent=None,  # Keep with current agent to analyze the content
            context_variables=shared_state.model_dump(),
        )

    def save_content_analysis(
        content_analysis: dict[str, Any] | ContentAnalysis | None = None,
        context_variables: dict = {},
    ) -> SwarmResult:
        """This function saves result of the agent getting the content analysis
        to the shared state."""
        if not context_variables:
            return SwarmResult(
                values="Error: Missing context_variables",
                agent=None,
                context_variables={},
            )

        shared_state = SharedContext.model_validate(context_variables)

        # If content_analysis was not provided directly, check if we have content to analyze
        if content_analysis is None:
            if shared_state.content:
                # Return this message to inform that analysis is needed
                return SwarmResult(
                    values="Content provided. Please analyze and provide a ContentAnalysis according to the defined schema.",
                    agent=None,  # Stay with the current agent to get proper analysis
                    context_variables=shared_state.model_dump(),
                )
            else:
                # We can't proceed without content and analysis
                return SwarmResult(
                    values="Error: Missing content. Please provide content to analyze.",
                    agent=None,
                    context_variables=shared_state.model_dump(),
                )

        if isinstance(content_analysis, ContentAnalysis):
            validated_analysis = content_analysis.model_dump()
        else:
            validated_analysis = ContentAnalysis.model_validate(
                content_analysis
            ).model_dump()

        shared_state.update_content_analysis(validated_analysis)

        return SwarmResult(
            values="Content analysis successfully validated and saved. Moving to next agent.",
            agent=next_agent,
            context_variables=shared_state.model_dump(),
        )

    agent = AssistantAgent(
        name="ContentAnalyzerAgent",
        system_message=system_msg.replace("${language}", language),
        llm_config=llm_config,
        functions=[save_original_content, save_content_analysis],
    )
    return agent
