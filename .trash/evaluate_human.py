#!/usr/bin/env python3
"""
Human evaluation script for RealBench tasks.
Allows humans to manually solve tasks and record their performance.
"""

import argparse
import logging

# Import the human agent and core functionality
import realeval_agents  # This registers the agents
from realeval import create_agent_args, get_tasks, run_tasks, format_human_results

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Run human evaluation of RealBench tasks.")
    
    # Task selection
    task_group = parser.add_argument_group("Task Selection")
    task_group.add_argument(
        "--task_name",
        type=str,
        default="webclones.zilloft-1",
        help="Name of the task to run (takes precedence over task_type and task_id).",
    )
    task_group.add_argument(
        "--task_type", 
        help="Task type to filter (e.g., 'omnizon', 'dashdish')"
    )
    task_group.add_argument(
        "--task_id", 
        type=int, 
        help="Specific task ID to run (requires --task_type)"
    )
    
    # Special parameters
    special_group = parser.add_argument_group("Special Parameters")
    special_group.add_argument(
        "--start_url",
        type=str,
        default="https://www.google.com",
        help="Starting URL (only for the openended task).",
    )
    
    # Environment setup
    env_group = parser.add_argument_group("Environment Configuration")
    env_group.add_argument(
        "--golden_user_data_dir",
        type=str,
        default=None,
        help="Path to a user data directory to use as a golden profile.",
    )
    env_group.add_argument(
        "--extensions_dir",
        type=str,
        default=None,
        help="Path to a directory containing Chrome extensions to load.",
    )
    env_group.add_argument(
        "--viewport_width",
        type=int,
        default=1280,
        help="Width of the browser viewport.",
    )
    env_group.add_argument(
        "--viewport_height",
        type=int,
        default=720,
        help="Height of the browser viewport.",
    )
    env_group.add_argument(
        "--max_steps",
        type=int,
        default=100,
        help="Maximum number of steps per task.",
    )
    
    # Output configuration
    output_group = parser.add_argument_group("Output Configuration")
    output_group.add_argument(
        "--results_dir",
        type=str,
        default="./results",
        help="Directory to store results.",
    )
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Determine which tasks to run
    if args.task_name:
        # If a direct task name is provided, use it (taking precedence)
        tasks = [args.task_name]
    else:
        # Otherwise use task_type and task_id for filtering
        tasks = get_tasks(
            task_type=args.task_type,
            task_id=args.task_id,
        )
        
    if not tasks:
        print("No tasks found with the specified criteria.")
        return

    # Print which tasks will be run
    print(f"Running human evaluation on {len(tasks)} task(s):")
    for task in tasks:
        print(f"  {task}")
    
    # Create the human agent arguments
    agent_args = create_agent_args("human")
    
    # Set up environment configuration
    env_args = {
        "task_seed": None,
        "max_steps": args.max_steps,
        "headless": False,  # Always show browser for human evaluation
        "golden_user_data_dir": args.golden_user_data_dir,
        "extensions_dir": args.extensions_dir,
        "viewport": {"width": args.viewport_width, "height": args.viewport_height},
    }
    
    # Add special parameters for openended task
    if "openended" in tasks:
        env_args["task_kwargs"] = {"start_url": args.start_url}
    
    # Run the evaluation
    results = run_tasks(
        tasks=tasks,
        agent_args=agent_args,
        env_args_dict=env_args,
        results_dir=args.results_dir,
        parallel=False,  # Human evaluation must be sequential
    )
    
    # Format and display results
    format_human_results(results)

if __name__ == "__main__":
    main()