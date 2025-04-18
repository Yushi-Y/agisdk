#!/usr/bin/env python3
"""
Examples demonstrating the usage of the agisdk harness.
"""

import dataclasses
from typing import Dict, Tuple, Union, Optional

from agisdk import real


# Example using a built-in agent with a specified model
def run_builtin_agent():
    # Create a harness with the gpt-4o model
    harness = real.harness(
        model="gpt-4o",
        task_name="webclones.omnizon-2",  # Specific task
        headless=False,                   # Show browser window
        max_steps=25,                     # Maximum steps per task
        use_screenshot=True,              # Include screenshots in observations
        use_axtree=True,                  # Include accessibility tree
    )
    
    # Run the task and get results
    results = harness.run()
    return results


# Example creating and using a custom agent
def run_custom_agent():
    # Define a custom agent
    class MyCustomAgent(real.Agent):
        def __init__(self) -> None:
            super().__init__()
            self.steps = 0
            
        def get_agent_action(self, obs) -> Tuple[Optional[str], Optional[str]]:
            """
            Core agent logic - analyze observation and decide on action.
            
            Returns:
                Tuple of (action, final_message)
                - If action is None, episode ends with final_message
                - If action is not None, the agent takes the action and continues
            """
            input("Insert your agent logic here...")
            self.steps += 1
            
            # Example of simple decision making based on URL
            current_url = obs.get("url", "")
            
            # Example logic: Search for a product
            if "google.com" in current_url:
                return "goto('https://www.amazon.com')", None
            elif "amazon.com" in current_url and self.steps == 1:
                return "type('input[name=\"field-keywords\"]', 'wireless headphones')", None
            elif "amazon.com" in current_url and self.steps == 2:
                return "click('input[type=\"submit\"]')", None
            elif "amazon.com" in current_url and self.steps >= 3:
                # Complete the task with a message
                return None, "Found wireless headphones on Amazon!"
            else:
                return "goto('https://www.google.com')", None
        
        def get_action(self, obs: dict) -> Tuple[str, Dict]:
            """
            Convert agent's high-level action to browsergym action.
            This method is required by the browsergym interface.
            """
            agent_action, final_message = self.get_agent_action(obs)
            
            if final_message:
                # End the episode with a message
                return f"send_msg_to_user(\"{final_message}\")", {}
            else:
                # Continue with the specified action
                return agent_action, {}

    # Create agent arguments class
    @dataclasses.dataclass
    class MyCustomAgentArgs(real.AbstractAgentArgs):
        agent_name: str = "MyCustomAgent"
        
        def make_agent(self):
            return MyCustomAgent()

    # Create harness with custom agent
    harness = real.harness(
        agentargs=MyCustomAgentArgs(),
        task_name="webclones.omnizon-1",
        headless=False,
    )
    
    # Run the task
    results = harness.run()
    return results


# Example running multiple tasks with leaderboard submission
def run_leaderboard_submission():
    # Create a harness for leaderboard submission
    harness = real.harness(
        model="gpt-4o",
        leaderboard=True,
        run_id="my_run_id_123",    # Your unique run ID for the leaderboard
        task_type="omnizon",       # Run all omnizon tasks
        headless=True,             # Run headless for submissions
        parallel=True,             # Run tasks in parallel
        num_workers=4,             # Number of parallel workers
    )
    
    # Run tasks
    results = harness.run()
    return results


if __name__ == "__main__":
    # Run the built-in agent example
    print("Running built-in agent example...")
    results = run_custom_agent()
    
    # Uncomment to run other examples
    # print("\nRunning custom agent example...")
    # results = run_custom_agent()
    
    # print("\nRunning leaderboard submission example...")
    # results = run_leaderboard_submission()