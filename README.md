<p align="center">
  <a href="https://theagi.company">
    <h3 align="center">AGI SDK</h3>
  </a>
</p>

AGI SDK is a toolkit for building and evaluating AI agents. It comes with support for the [REAL benchmark](https://realevals.xyz) to evaluate browser-based agents in real world settings.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/agisdk.git
cd agisdk

# Install in development mode
pip install -e .
```

## Quick Start

Here's a simple example to get you started:

```python
from agisdk import real

# Create a harness with a pre-configured model
harness = real.harness(
    model="gpt-4o",
    task_name="webclones.omnizon-1",
    headless=False
)

# Run the experiment
results = harness.run()
```

## Creating Custom Agents

You can create your own agent by extending the Agent class:

```python
import dataclasses
from agisdk import real
from agisdk.real.browsergym.experiments import Agent, AbstractAgentArgs
from typing import Dict, Tuple

class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__()
        # Initialize your agent's state

    def get_agent_action(self, obs) -> Tuple[str, str]:
        """
        Core agent logic to analyze observations and select actions.

        Args:
            obs: The observation from the environment

        Returns:
            A tuple of (action_string, final_message)
            - If final_message is None, the episode continues with the given action
            - If action_string is None and final_message is provided, the episode ends
        """
        # Example: Navigate to Google
        if "google" not in obs.get("url", ""):
            return "goto('https://www.google.com')", None

        # Example: End the episode
        return None, "Task completed successfully!"

    def get_action(self, obs: dict) -> Tuple[str, Dict]:
        """
        Convert agent_action output to the format expected by browsergym.
        You usually don't need to modify this method.
        """
        agent_action, final_message = self.get_agent_action(obs)

        if final_message:
            return f"send_msg_to_user(\"{final_message}\")", {}
        else:
            return agent_action, {}

@dataclasses.dataclass
class MyAgentArgs(AbstractAgentArgs):
    agent_name: str = "MyAgent"

    def make_agent(self):
        return MyAgent()

# Create a harness with your custom agent
harness = real.harness(
    agentargs=MyAgentArgs(),
    task_name="webclones.omnizon-1",
    headless=False
)

# Run the experiment
results = harness.run()
```

## Observation Structure

Your agent gets access to the following observation structure:

```python
{
    'chat_messages': [...],          # History of chat messages
    'goal': "...",                   # Text description of the goal
    'goal_object': [...],            # Structured goal object with text and images
    'open_pages_urls': [...],        # List of open page URLs
    'active_page_index': 0,          # Index of the active page
    'url': "...",                    # Current URL
    'screenshot': np.array(...),     # Screenshot as numpy array
    'dom_object': {...},             # DOM structure
    'axtree_object': {...},          # Accessibility tree
    'extra_element_properties': {...}, # Additional element properties
    'focused_element_bid': "...",    # ID of the focused element
    'last_action': "...",            # Last action performed
    'last_action_error': "...",      # Error from last action (if any)
    'elapsed_time': 0.0,             # Time elapsed in the episode
    'browser': {...}                 # Browser object
}
```

## Available Actions

Actions are specified as strings in the format of function calls. Here are some commonly used actions:

```python
# Navigation
"goto('https://www.google.com')"
"go_back()"
"go_forward()"

# Interaction
"click('element_id')"
"fill('input_id', 'text to enter')"
"press('Enter')"

# Communication
"send_msg_to_user('I found the answer: $42.99')"

# Reporting infeasible tasks
"report_infeasible('The requested item is out of stock')"
```

## Available Tasks

The AGISDK includes several pre-configured task environments:

- `webclones.omnizon-1` through `webclones.omnizon-10`: E-commerce shopping tasks
- `webclones.dashdish-1` through `webclones.dashdish-11`: Food delivery tasks
- `webclones.fly-unified-1` through `webclones.fly-unified-14`: Flight booking tasks
- And many more...

## Harness Configuration

The harness function accepts the following parameters:

```python
real.harness(
    # Agent configuration (provide one of these)
    model="gpt-4o",           # Use the pre-configured DemoAgent with this model
    agentargs=MyAgentArgs(),  # Or provide your own agent arguments

    # Task configuration
    task_name="webclones.omnizon-1",  # Which task to run
    headless=False,                   # Whether to show the browser
    wait_for_user_message=False,      # Whether to wait for user messages
    max_steps=100,                    # Maximum number of steps

    # Tracking
    leaderboard=False,       # Whether to submit to leaderboard (WIP)
    run_id="my_unique_id",   # Unique ID for the run

    # Output
    results_dir="./results"   # Where to store results
)
```

## Example Scripts

Check out the example scripts in the repository:

- `harness_example.py`: Basic usage of the harness
- `example.py`: More advanced example with custom agent logic

## Contributing

We welcome any contributions to the AGI SDK, whether it's submitting an idea, fixing a typo, adding a new guide, or improving an existing one.

If you have ideas for new examples or guides, share them on the [issues page](https://github.com/agi-inc/agisdk/issues).

If you want to directly contribute code, you can fork the repository, make your changes, and submit a pull request.
To avoid duplication of efforts, please review the existing issues and pull requests before contributing.

Happy building! 🙌
