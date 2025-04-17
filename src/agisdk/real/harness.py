"""
Real Browser Gym Harness

This module provides a simple harness for running agents in browsergym environments.
"""

import os
import logging
from typing import Optional, Union
import uuid
from pathlib import Path

from agisdk.real.browsergym.experiments import EnvArgs, ExpArgs
# No need to import run, we'll use exp_args.run() directly
from agisdk.real.demo_agent.basic_agent import DemoAgentArgs
from agisdk.real.browsergym.experiments import AbstractAgentArgs

logger = logging.getLogger(__name__)

def harness(
    model: Optional[str] = None,
    agentargs: Optional[AbstractAgentArgs] = None,
    leaderboard: bool = False,
    run_id: Optional[str] = None,
    task_name: str = "webclones.omnizon-1",
    headless: bool = False,
    wait_for_user_message: bool = False,
    max_steps: int = 25,
    results_dir: str = "./results",
    **kwargs
):
    """
    Create a harness for running agents in browsergym environments.
    
    Args:
        model: The model name to use with DemoAgent if no agent args are provided.
        agentargs: Agent arguments object. If provided, overrides model.
        leaderboard: Whether to submit results to the leaderboard.
        run_id: Unique ID for this run. If not provided, one will be generated.
        task_name: The name of the task to run.
        headless: Whether to run in headless mode.
        wait_for_user_message: Whether to wait for user messages in chat mode.
        max_steps: Maximum number of steps to run.
        results_dir: Directory to store results.
        **kwargs: Additional arguments to pass to the environment.
        
    Returns:
        A harness object for running agents in browsergym environments.
    """
    # Setup agent args
    if agentargs is None and model is not None:
        # If only model is specified, use DemoAgent
        agent_args = DemoAgentArgs(
            model_name=model,
            chat_mode=wait_for_user_message,
            demo_mode="default" if not headless else "off",
            use_html=False,
            use_axtree=True,
            use_screenshot=True,
        )
    elif agentargs is not None:
        # If agentargs is specified, use that
        agent_args = agentargs
    else:
        raise ValueError("Either model or agentargs must be specified")

    # Generate run_id if not provided
    if run_id is None:
        run_id = str(uuid.uuid4())
    
    # Setup environment args
    env_args = EnvArgs(
        task_name=task_name,
        task_seed=None,
        max_steps=max_steps,
        headless=headless,
        wait_for_user_message=wait_for_user_message,
        **kwargs
    )
    
    # Setup experiment args
    exp_args = ExpArgs(
        env_args=env_args,
        agent_args=agent_args,
    )
    
    # Return an object with a run method
    class Harness:
        def __init__(self, exp_args, results_dir, run_id, leaderboard):
            self.exp_args = exp_args
            self.results_dir = str(Path(results_dir).absolute())
            self.run_id = run_id
            self.leaderboard = leaderboard
        
        def run(self):
            """Run the experiment and return the results."""
            logger.info(f"Running experiment with ID: {self.run_id}")
            
            # Prepare the experiment directory
            self.exp_args.prepare(self.results_dir)
            
            # Run the experiment
            self.exp_args.run()
            
            # TODO: If leaderboard is True, submit results
            if self.leaderboard:
                logger.info("Leaderboard submission is not yet implemented")
            
            # Return None for now, could return results summary in the future
            return None
    
    return Harness(exp_args, results_dir, run_id, leaderboard)