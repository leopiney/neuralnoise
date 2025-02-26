# Neural Noise: Agent Context Sharing Workflow

This document outlines how context is shared between agents in the Neural Noise podcast generation system.

## Overview

The podcast generation system employs a multi-agent approach where different specialized agents collaborate to transform raw content into a structured podcast script. Each agent has a specific role, but all agents need to share information through a common context.

## Context Sharing Architecture

### 1. SharedContext Class

The central piece of context sharing is the `SharedContext` class, which provides:

- A structured model for storing all shared state
- Methods for accessing and updating specific parts of the context
- Error and warning tracking capabilities
- Type-safe interface with validation

```python
class SharedContext(BaseModel):
    content: Optional[str]  # Raw content being processed
    content_analysis: Optional[Dict[str, Any]]  # Analysis results
    sections: List[Dict[str, Any]]  # Content sections metadata
    section_scripts: Dict[int, Dict[str, Any]]  # Section scripts
    current_section_index: int  # Active section index
    is_complete: bool  # Processing completion flag
    errors: List[str]  # Error messages
    warnings: List[str]  # Warning messages

    # Methods for manipulating the context...
```

### 2. AG2 Swarm Context Mechanism

Agents in the system leverage AG2's swarm context mechanism:

1. Context is passed through `SwarmResult` objects between agent function calls
2. Context is accessed by agents via `agent.get_context()`
3. Context is propagated through the handoff chain to subsequent agents

## Agent Context Flow

The workflow follows this pattern:

1. **UserProxyAgent** initiates with raw content
2. **ContentAnalyzerAgent**
   - Stores raw content in context
   - Analyzes content and stores analysis in context
   - Passes context to PlannerAgent
3. **PlannerAgent**
   - Reads content analysis from context
   - Plans sections and adds them to context
   - Passes context to ScriptGeneratorAgent
4. **ScriptGeneratorAgent**
   - Reads current section from context
   - Generates script for current section
   - Updates section script in context
   - Passes context to EditorAgent
5. **EditorAgent**
   - Reviews script from context
   - Decides next agent based on context state
   - Updates section index in context if moving to next section
   - Passes context to appropriate next agent

## Context Access Patterns

### 1. Function Parameters

Agent functions receive context as a parameter:

```python
def write_section_script(context_variables: Dict[str, Any]) -> SwarmResult:
    # Convert dict to model for type-safety and validation
    shared_state = SharedContext.model_validate(context_variables)

    # Access context
    current_section = shared_state.get_current_section()

    # Update context
    shared_state.update_section_script(script.model_dump())

    # Return updated context
    return SwarmResult(
        values=json.dumps(script.model_dump()),
        agent=next_agent,
        context_variables=shared_state.model_dump(),
    )
```

### 2. Agent Context API

Agents can access context directly:

```python
def update_system_message(agent, messages) -> str:
    # Get context from agent
    context = agent.get_context()
    shared_state = SharedContext.model_validate(context)

    # Use context to customize system message
    current_section = shared_state.get_current_section()
    # ...
```

### 3. Error & Warning Handling

```python
# Record errors in context
shared_state.add_error("Missing section data")

# Record warnings in context
shared_state.add_warning("Non-critical issue detected")

# Handle errors appropriately
if shared_state.has_errors():
    # Take remedial action
```

## Best Practices

1. **Consistent Access Pattern**: Always convert dictionary context to `SharedContext` model
2. **Error Recording**: Record errors and warnings in context
3. **Context Propagation**: Always return updated context in `SwarmResult`
4. **State Validation**: Use Pydantic validation to ensure context integrity
5. **Minimal Context**: Only store necessary information in context

## Handoff Registration

Agents are registered for handoffs using the AG2 `register_hand_off` function:

```python
# ContentAnalyzerAgent -> PlannerAgent
register_hand_off(
    content_analyzer_agent,
    [
        OnCondition(
            planner_agent,
            "After content analysis, transfer to planning"
        )
    ],
)
```

This ensures context flows through the system in a controlled manner.

## Context Debugging

The system provides methods to inspect context state:

1. Error and warning logs are maintained in the context
2. Each agent can access the full context history
3. Errors are propagated through the context to aid debugging

## Conclusion

The context sharing mechanism in Neural Noise ensures:

1. Seamless information flow between specialized agents
2. Type-safe and validated context data
3. Proper error tracking and propagation
4. Clear separation of concerns between agents
5. Flexible workflow control through context-aware handoffs
