#!/usr/bin/env python3
"""
Agent implementations for RealBench evaluation.
"""

import json
import os
import dataclasses
from typing import Dict, Tuple

from agisdk.REAL.browsergym.experiments import Agent, AbstractAgentArgs
from agisdk.REAL.demo_agent.basic_agent import DemoAgentArgs
from operator_agent import OperatorAgentArgs
# from custom_agent import your_agent

# Register this module's agents with the realeval registry
from realeval import register_agent

class HumanAgent(Agent):
    """A simple agent that lets the human interact directly with the browser."""

    def __init__(self) -> None:
        super().__init__()
        self.answer_provided = False
        self.user_answer = None

    def obs_preprocessor(self, obs: dict) -> dict:
        """Pass through observations unmodified."""
        return obs

    def get_action(self, obs: dict) -> Tuple[str, Dict]:
        """Get action from human input or wait if already provided."""
        # First time the agent is called, inform the user what to do
        if not self.answer_provided:
            print("Solve the task in the browser window. When finished, enter your answer below.")
            user_answer = input("Your answer: ")
            self.user_answer = str(user_answer)
            self.answer_provided = True
            escaped_answer = json.dumps(self.user_answer)
            return f"send_msg_to_user({escaped_answer})", {}
        else:
            # On subsequent calls, wait until user indicates they're done
            input("Press Enter when you're finished with the task...")
            return "wait(1000)", {}


@dataclasses.dataclass
class HumanAgentArgs(AbstractAgentArgs):
    """Arguments for the human agent."""
    agent_name: str = "Human"
    
    def make_agent(self):
        return HumanAgent()

# Custom Browser Control Agent
class CustomAgent(Agent):
    """Agent that provides direct browser control using the custom_agent module."""

    def __init__(self) -> None:
        super().__init__()
        self.action_sent = False
        self.user_answer = None

    def obs_preprocessor(self, obs: dict) -> dict:
        """Pass through observations unmodified."""
        return obs

    def get_action(self, obs: dict) -> Tuple[str, Dict]:
        """Give browser control to custom_agent on first call."""
        if not self.action_sent:
            # Get the browser object
            browser = obs.get("browser")
            if browser:
                # Run the custom agent
                result = your_agent(browser)
                self.user_answer = result
                self.action_sent = True
                escaped_answer = json.dumps(self.user_answer)
                return f"send_msg_to_user({escaped_answer})", {}
            else:
                print("Error: Browser object not available in observation")
                return "wait(1000)", {}
        else:
            # On subsequent calls, just wait
            return "wait(1000)", {}


@dataclasses.dataclass
class CustomAgentArgs(AbstractAgentArgs):
    """Arguments for the custom browser agent."""
    agent_name: str = "Custom"
    
    def make_agent(self):
        return CustomAgent()


# Register agents with the registry
register_agent("human", HumanAgentArgs)
register_agent("ai", DemoAgentArgs)
register_agent("operator", OperatorAgentArgs)
register_agent("custom", CustomAgentArgs)
