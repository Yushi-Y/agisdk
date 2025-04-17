#!/usr/bin/env python3
"""
Example of using the harness to run agents in browsergym environments.
"""

import dataclasses
from agisdk import real
from agisdk.real.browsergym.experiments import Agent, AbstractAgentArgs
from typing import Dict, Tuple

# Option 1: Using the demo agent with a specified model
harness1 = real.harness(
    model="gpt-4o", 
    task_name="webclones.omnizon-2",
    headless=False,
    max_steps=12,
)

# Option 2: Creating and using a custom agent
class YourAgent(Agent):
    def __init__(self) -> None:
        super().__init__()
        self.answer_provided = False
        self.user_answer = None
        
    def get_agent_action(self, obs) -> Tuple[str, str]:
        # Your agent logic goes here
        # In a real implementation, analyze the observation and decide on an action
        
        # Example: Return an action without ending the episode
        if "google" not in obs.get("url", ""):
            return "goto('https://www.google.com')", None
            
        # Example: End the episode with a message
        return None, "Task completed Successfully!"
    
    def get_action(self, obs: dict) -> Tuple[str, Dict]:
        agent_action, final_message = self.get_agent_action(obs)
        
        if final_message:
            return f"send_msg_to_user(\"{final_message}\")", {}
        else:
            return agent_action, {}

@dataclasses.dataclass
class YourAgentArgs(AbstractAgentArgs):
    agent_name: str = "YourAgent"
    
    def make_agent(self):
        return YourAgent()

# Create a harness with your custom agent
harness2 = real.harness(
    agentargs=YourAgentArgs(),
    leaderboard=False,
    task_name="webclones.omnizon-1",
    headless=False
)

# Run the experiment with the demo agent
# Uncomment to run
# results1 = harness1.run()

# Run the experiment with your custom agent
# Uncomment to run
# results2 = harness2.run()

if __name__ == "__main__":
    # Choose which harness to run
    print("Running demo agent...")
    results = harness1.run()
    
    # Print a summary of the results (currently None)
    print("Results:", results)