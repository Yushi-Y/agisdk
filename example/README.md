# AGI SDK Examples

This directory contains example implementations demonstrating how to use the AGI SDK for building and testing web agents.

## Available Examples

### 1. `example.py`

A basic example demonstrating the core functionality of the AGI SDK harness with both built-in and custom agents.

**Features:**
- Using a built-in agent with a specified model
- Creating and using a custom agent
- Running multiple tasks with leaderboard submission

**How to run:**
```bash
# Run the default example (built-in agent)
python example.py

# To run the custom agent example, modify the main block to call run_custom_agent()
```

### 2. `hackable_agent.py`

A more advanced example with a customizable agent that can be modified for different use cases.

**Features:**
- Detailed agent implementation with observation preprocessing
- Support for different model backends (OpenAI, Anthropic, OpenRouter)
- Configurable agent parameters

**How to run:**
```bash
# Run with the default configuration
python hackable_agent.py

# To run with custom agent or leaderboard submission, modify the main block
# to call run_custom_agent() or run_leaderboard_submission()
```

## Harness Configuration

Both examples use the REAL harness, which accepts various configuration parameters:

```python
REAL.harness(
    # Agent configuration (provide one of these)
    model="gpt-4o",                                # OpenAI models
    model="sonnet-3.7",                            # Anthropic models
    model="openrouter/deepseek/deepseek-chat-v3-0324", # OpenRouter models
    agentargs=MyAgentArgs(),                       # Or custom agent arguments

    # Task selection (provide one of these)
    task_name="webclones.omnizon-1",  # Specific task to run
    task_type="omnizon",              # Run all tasks of this type
    task_id=1,                        # Run specific task ID within a type

    # Browser configuration
    headless=False,                   # Whether to show the browser
    max_steps=25,                     # Maximum number of steps
    
    # Observation options
    use_html=False,                   # Include HTML in observations
    use_axtree=True,                  # Include accessibility tree
    use_screenshot=True,              # Include screenshots

    # Leaderboard submission
    leaderboard=False,                # Whether to submit to leaderboard
    run_id="my_unique_id",            # Unique ID for the submission

    # Execution options
    parallel=False,                   # Run tasks in parallel
    num_workers=4,                    # Number of parallel workers
)
```

## Creating Your Own Agent

To create your own agent:

1. Subclass `REAL.Agent` and implement the required methods:
   - `__init__`: Initialize your agent
   - `get_action`: Process observations and return actions

2. Create an arguments class by subclassing `REAL.AbstractAgentArgs`

3. Use your agent with the harness:
   ```python
   harness = REAL.harness(
       agentargs=YourCustomAgentArgs(),
       # other configuration options
   )
   ```

See the `MyCustomAgent` class in `example.py` or the `DemoAgent` class in `hackable_agent.py` for implementation examples.
