#!/usr/bin/env python3
"""
Interactive guided evaluation session for RealBench tasks.
Walks users through agent selection and task execution with a step-by-step approach.
"""

import argparse
import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any

# Import the agent and core functionality 
import realeval_agents  # This registers the agents
# Make sure webclones tasks are registered
from agisdk.REAL.browsergym import webclones
from realeval import (
    create_agent_args, 
    get_tasks, 
    run_tasks, 
    format_benchmark_results,
    format_human_results,
    get_available_task_types
)
from operator_agent import OperatorAgentArgs

logger = logging.getLogger(__name__)

class GuidedRealBenchSession:
    """Interactive guided session for RealBench evaluation."""
    
    def __init__(self, args):
        """Initialize with commandline arguments."""
        self.args = args
        self.session_results = {}
        self.last_agent_type = None
        self.last_task = None
        
        # Add parallelism attributes if not present
        if not hasattr(args, "parallel"):
            args.parallel = False
        if not hasattr(args, "num_workers"):
            args.num_workers = 4
        
        # Initialize environment arguments
        self.env_args = {
            "task_seed": None,
            "max_steps": args.max_steps,
            "headless": False,  # Always interactive for guided sessions
            "golden_user_data_dir": args.golden_user_data_dir,
            "extensions_dir": args.extensions_dir,
            "viewport": {"width": args.viewport_width, "height": args.viewport_height},
        }
        
        # Add task_kwargs with run_id if provided
        if hasattr(args, "run_id") and args.run_id:
            self.env_args["task_kwargs"] = {"run_id": args.run_id}
        
        # Create default results directory if it doesn't exist
        if not os.path.exists(args.results_dir):
            os.makedirs(args.results_dir)
        
        # Cache these for reuse
        self.available_task_types = get_available_task_types()
        print(self.available_task_types)
    
    def start(self):
        """Start the guided session."""
        print("\n" + "="*50)
        print("Welcome to the RealBench Guided Evaluation Session")
        print("="*50)
        
        try:
            while True:
                print("\nWhat would you like to do?")
                print("1. Run a task")
                print("2. View session results")
                print("3. Change settings")
                print("4. Save/load session")
                print("5. Quick actions")
                print("6. Exit")
                
                choice = input("\nEnter your choice (1-6): ").strip()
                
                if choice == "1":
                    self._guided_run_task()
                elif choice == "2":
                    self._view_results()
                elif choice == "3":
                    self._change_settings()
                elif choice == "4":
                    self._save_load_menu()
                elif choice == "5":
                    self._quick_actions_menu()
                elif choice == "6":
                    self._save_session()
                    print("\nGoodbye!")
                    break
                else:
                    print("Invalid choice. Please select a number from 1-6.")
        except KeyboardInterrupt:
            print("\n\nSession interrupted. Saving results...")
            self._save_session()
            print("Goodbye!")
    
    def _guided_run_task(self):
        """Guide the user through selecting and running a task."""
        # Step 1: Choose agent type
        print("\n-- Choose an Agent --")
        print("1. Human (you'll interact with the browser)")
        print("2. AI (an AI agent will perform the task)")
        print("3. Operator (openai agent)")
        print("4. Custom (direct browser control)")
        
        agent_choice = input("\nSelect agent type (1-4): ").strip()
        if agent_choice not in ["1", "2", "3", "4"]:
            print("Invalid choice. Returning to main menu.")
            return
            
        if agent_choice == "1":
            agent_type = "human"
        elif agent_choice == "2":
            agent_type = "ai"
        elif agent_choice == "3":
            agent_type = "operator"
        else:  # agent_choice == "4"
            agent_type = "custom"
        
        # Step 2: Choose task type or all types
        print("\n-- Choose Task Type --")
        print("Available task types:")
        
        for i, task_type in enumerate(self.available_task_types, 1):
            print(f"{i}. {task_type}")
        print(f"{len(self.available_task_types) + 1}. All task types")
        
        task_type_choice = input(f"\nSelect task type (1-{len(self.available_task_types) + 1}): ").strip()
        
        try:
            task_type_idx = int(task_type_choice) - 1
            if task_type_idx == len(self.available_task_types):
                # Selected "All task types"
                selected_task_type = None
            elif 0 <= task_type_idx < len(self.available_task_types):
                selected_task_type = self.available_task_types[task_type_idx]
            else:
                print("Invalid choice. Returning to main menu.")
                return
        except ValueError:
            print("Invalid input. Returning to main menu.")
            return
        
        # Step 3: Get tasks for the selected type
        available_tasks = get_tasks(task_type=selected_task_type)
        if not available_tasks:
            print(f"No tasks found for the selected criteria.")
            return
        
        # Step 4: Choose specific task or all tasks of that type
        print("\n-- Choose Task --")
        
        # Strip "webclones." prefix for display
        display_tasks = [t.replace("webclones.", "") for t in available_tasks]
        
        # Display tasks with numbers
        for i, task in enumerate(display_tasks, 1):
            print(f"{i}. {task}")
        print(f"{len(available_tasks) + 1}. Run all {len(available_tasks)} tasks")
        
        task_choice = input(f"\nSelect task (1-{len(available_tasks) + 1}): ").strip()
        
        try:
            task_idx = int(task_choice) - 1
            if task_idx == len(available_tasks):
                # Selected "Run all tasks"
                selected_tasks = available_tasks
            elif 0 <= task_idx < len(available_tasks):
                selected_tasks = [available_tasks[task_idx]]
            else:
                print("Invalid choice. Returning to main menu.")
                return
        except ValueError:
            print("Invalid input. Returning to main menu.")
            return
        
        # Create agent arguments
        try:
            if agent_type == "human":
                agent_args = create_agent_args("human")
            elif agent_type == "ai":
                system_message_handling = "combined" if self.args.model == "o1-mini" else "separate"
                agent_args = create_agent_args("ai", 
                    model_name=self.args.model,
                    chat_mode=False,
                    demo_mode="default",
                    use_html=self.args.use_html,
                    use_axtree=self.args.use_axtree,
                    use_screenshot=self.args.use_screenshot,
                    system_message_handling=system_message_handling
                )
            elif agent_type == "operator":
                # Operator agent manages its own model and feature usage internally
                # We only need to pass the type. Dimensions are passed later via env_args.
                agent_args = create_agent_args("operator")
            elif agent_type == "custom":
                # Custom agent provides direct browser control
                agent_args = create_agent_args("custom")
            else:
                print("Invalid agent type. Returning to main menu.")
                return
        except Exception as e:
            print(f"Error creating agent: {e}")
            return
        
        # Run the selected task(s)
        plural = "s" if len(selected_tasks) > 1 else ""
        print(f"\nRunning {len(selected_tasks)} task{plural} with {agent_type} agent...")
        
        try:
            # Determine whether to use parallelism (only for multiple tasks)
            use_parallel = getattr(self.args, "parallel", False) and len(selected_tasks) > 1
            num_workers = getattr(self.args, "num_workers", 4) if use_parallel else 1
            
            # Get caching settings from args or use defaults
            use_cache = getattr(self.args, "use_cache", True)
            cache_only = getattr(self.args, "cache_only", False)
            force_refresh = getattr(self.args, "force_refresh", False)
            
            # Don't cache human results
            if agent_type == "human":
                use_cache = False
            
            # Run with or without parallelism
            results = run_tasks(
                tasks=selected_tasks,
                agent_args=agent_args,
                env_args_dict=self.env_args,
                results_dir=self.args.results_dir,
                parallel=use_parallel,
                num_workers=num_workers,
                use_cache=use_cache,
                cache_only=cache_only,
                force_refresh=force_refresh
            )
            
            # Store results
            self.session_results.update(results)
            
            # Remember last selections
            self.last_agent_type = agent_type
            self.last_task = selected_tasks[0] if len(selected_tasks) == 1 else None
            
            # Format and display results
            if agent_type == "human":
                format_human_results(results)
            else:
                format_benchmark_results(results)
            
            # Save session after each run
            self._save_session()
            
            # Option to repeat or switch agent
            if len(selected_tasks) == 1:
                self._offer_repeat_options(selected_tasks[0])
            
        except Exception as e:
            print(f"Error running task(s): {e}")
    
    def _offer_repeat_options(self, task_name):
        """Offer options to repeat the task or switch agent types."""
        print("\nWould you like to:")
        print("1. Repeat this task with the same agent")
        print("2. Run this task yourself")
        print("3. Return to the main menu")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            self._run_single_task(self.last_agent_type, task_name)
        elif choice == "2":
            self._run_single_task("human", task_name)
        else:
            # For any other choice, just return to main menu
            print("Returning to main menu.")
    
    def _run_single_task(self, agent_type, task_name):
        """Run a single task with the specified agent type."""
        try:
            if agent_type == "human":
                agent_args = create_agent_args("human")
            elif agent_type == "ai":
                system_message_handling = "combined" if self.args.model == "o1-mini" else "separate"
                agent_args = create_agent_args("ai", 
                    model_name=self.args.model,
                    chat_mode=False,
                    demo_mode="default",
                    use_html=self.args.use_html,
                    use_axtree=self.args.use_axtree,
                    use_screenshot=self.args.use_screenshot,
                    system_message_handling=system_message_handling
                )
            elif agent_type == "operator":
                 # Operator agent manages its own model and feature usage internally
                 agent_args = create_agent_args("operator")
            elif agent_type == "custom":
                 # Custom agent provides direct browser control
                 agent_args = create_agent_args("custom")
            else:
                print(f"Unknown agent type '{agent_type}' for running single task.")
                return

            print(f"Running task {task_name} with {agent_type} agent...")

            # Get caching settings from args or use defaults
            use_cache = getattr(self.args, "use_cache", True)
            cache_only = getattr(self.args, "cache_only", False)
            force_refresh = getattr(self.args, "force_refresh", False)

            # Don't cache human results
            if agent_type == "human":
                use_cache = False

            results = run_tasks(
                tasks=[task_name],
                agent_args=agent_args,
                env_args_dict=self.env_args,
                results_dir=self.args.results_dir,
                parallel=False, # Always run single task sequentially
                use_cache=use_cache,
                cache_only=cache_only,
                force_refresh=force_refresh
            )

            # Store results and last selections
            self.session_results.update(results)
            self.last_agent_type = agent_type
            self.last_task = task_name

            # Format and display results
            if agent_type == "human":
                format_human_results(results)
            else:
                format_benchmark_results(results)

            # Save session
            self._save_session()

            # Offer repeat options again
            self._offer_repeat_options(task_name)

        except Exception as e:
            print(f"Error running task: {e}")
    
    def _view_results(self):
        """View session results."""
        if not self.session_results:
            print("\nNo results in this session yet.")
            return
        
        print("\n-- Session Results --")
        
        # Group results by task type
        task_type_results = {}
        for task_name, record in self.session_results.items():
            # Extract task type (e.g., "omnizon" from "webclones.omnizon-1" or "fly-unified" from "webclones.fly-unified-1")
            task_parts = task_name.split('.')
            if len(task_parts) > 1:
                task_full_name = task_parts[1]
                parts = task_full_name.split('-')
                
                # Find where the numeric part starts
                for i, part in enumerate(parts[1:], 1):
                    if part and part[0].isdigit():
                        task_type = '-'.join(parts[:i])
                        break
                else:
                    # Fallback if no numeric part is found
                    task_type = parts[0]
            else:
                task_type = "other"
                
            if task_type not in task_type_results:
                task_type_results[task_type] = []
                
            task_type_results[task_type].append((task_name, record))
        
        # Display results by task type
        for task_type, results in sorted(task_type_results.items()):
            print(f"\n{task_type.capitalize()} Tasks:")
            
            for task_name, record in results:
                agent_type = "AI" if "agent_name" in record and record["agent_name"] == "DemoAgent" else "Human"
                success = "✓" if record.get("cum_reward", 0) == 1 else "✗"
                print(f"  {task_name.replace('webclones.', '')} ({agent_type}): {success} Time={record.get('elapsed_time', 0):.2f}s")
        
        # Summary statistics
        total_tasks = len(self.session_results)
        successful_tasks = sum(1 for _, r in self.session_results.items() if r.get("cum_reward", 0) == 1)
        success_rate = (successful_tasks / total_tasks) * 100 if total_tasks > 0 else 0
        
        print(f"\nSummary: {successful_tasks}/{total_tasks} tasks completed successfully ({success_rate:.1f}%)")
        
        input("\nPress Enter to return to the main menu...")
    
    def _change_settings(self):
        """Menu for changing various settings."""
        while True:
            print("\n-- Change Settings --")
            print("1. Change AI model")
            print("2. Set maximum steps")
            print("3. Toggle features (HTML/screenshots/accessibility)")
            print("4. Set results directory")
            print("5. Browser and parallelism settings")
            print("6. Extensions and browser state")
            print("7. Caching settings")
            print("8. Set run ID for leaderboard submissions")
            print("9. Return to main menu")
            
            choice = input(f"\nEnter your choice (1-9): ").strip()
            
            if choice == "1":
                new_model = input(f"Enter new model name [{self.args.model}]: ").strip()
                if new_model:
                    self.args.model = new_model
                    print(f"Model set to: {new_model}")
                    
            elif choice == "2":
                try:
                    new_max_steps = input(f"Enter maximum steps [{self.args.max_steps}]: ").strip()
                    if new_max_steps:
                        self.args.max_steps = int(new_max_steps)
                        self.env_args["max_steps"] = int(new_max_steps)
                        print(f"Maximum steps set to: {new_max_steps}")
                except ValueError:
                    print("Invalid value. Please enter a number.")
                    
            elif choice == "3":
                print("\nFeature Settings:")
                print(f"1. Include HTML: {'Yes' if self.args.use_html else 'No'}")
                print(f"2. Include accessibility tree: {'Yes' if self.args.use_axtree else 'No'}")
                print(f"3. Include screenshots: {'Yes' if self.args.use_screenshot else 'No'}")
                
                feature_choice = input("\nEnter feature to toggle (1-3): ").strip()
                
                if feature_choice == "1":
                    self.args.use_html = not self.args.use_html
                    print(f"Include HTML set to: {'Yes' if self.args.use_html else 'No'}")
                elif feature_choice == "2":
                    self.args.use_axtree = not self.args.use_axtree
                    print(f"Include accessibility tree set to: {'Yes' if self.args.use_axtree else 'No'}")
                elif feature_choice == "3":
                    self.args.use_screenshot = not self.args.use_screenshot
                    print(f"Include screenshots set to: {'Yes' if self.args.use_screenshot else 'No'}")
                else:
                    print("Invalid choice.")
                    
            elif choice == "4":
                new_dir = input(f"Enter results directory [{self.args.results_dir}]: ").strip()
                if new_dir:
                    self.args.results_dir = new_dir
                    # Create directory if it doesn't exist
                    if not os.path.exists(new_dir):
                        os.makedirs(new_dir)
                    print(f"Results directory set to: {new_dir}")
            
            elif choice == "5":
                self._browser_and_parallelism_settings()
                
            elif choice == "6":
                self._extensions_and_browser_state_settings()
                
            elif choice == "7":
                self._caching_settings()
                    
            elif choice == "8":
                current_run_id = getattr(self.args, "run_id", None)
                current_display = current_run_id if current_run_id else "Not set"
                new_run_id = input(f"Enter run ID for leaderboard submissions [{current_display}]: ").strip()
                if new_run_id:
                    self.args.run_id = new_run_id
                    # Update in env_args
                    if "task_kwargs" not in self.env_args:
                        self.env_args["task_kwargs"] = {}
                    self.env_args["task_kwargs"]["run_id"] = new_run_id
                    print(f"Run ID set to: {new_run_id}")
                elif new_run_id == "" and current_run_id:  # User entered empty string
                    self.args.run_id = None
                    if "task_kwargs" in self.env_args and "run_id" in self.env_args["task_kwargs"]:
                        del self.env_args["task_kwargs"]["run_id"]
                    print("Run ID cleared")
                    
            elif choice == "9":
                return
                
            else:
                print("Invalid choice. Please select a number from 1-9.")
    
    def _browser_and_parallelism_settings(self):
        """Configure browser visibility and parallelism settings."""
        print("\n-- Browser and Parallelism Settings --")
        
        # Configure headless mode (browser visibility)
        current_headless = self.env_args.get("headless", False)
        print(f"1. Browser visibility: {'Hidden (headless)' if current_headless else 'Visible'}")
        
        # Configure parallelism
        current_parallel = getattr(self.args, "parallel", False)
        workers = getattr(self.args, "num_workers", 4)
        print(f"2. Run tasks in parallel: {'Yes' if current_parallel else 'No'}")
        if current_parallel:
            print(f"3. Number of parallel workers: {workers}")
        else:
            print("3. Number of parallel workers: N/A (parallelism disabled)")
            
        print("4. Return to settings menu")
        
        sub_choice = input("\nEnter setting to change (1-4): ").strip()
        
        if sub_choice == "1":
            new_visibility = input(f"Show browser? (yes/no) [{'no' if current_headless else 'yes'}]: ").strip().lower()
            if new_visibility:
                headless_mode = new_visibility not in ["y", "yes", "true", "1", "show"]
                self.env_args["headless"] = headless_mode
                if hasattr(self.args, "headless"):
                    self.args.headless = headless_mode
                print(f"Browser will be {'hidden' if headless_mode else 'visible'}")
        
        elif sub_choice == "2":
            new_parallel = input(f"Run tasks in parallel? (yes/no) [{'yes' if current_parallel else 'no'}]: ").strip().lower()
            if new_parallel:
                parallel_mode = new_parallel in ["y", "yes", "true", "1"]
                if not hasattr(self.args, "parallel"):
                    self.args.parallel = parallel_mode
                else:
                    self.args.parallel = parallel_mode
                print(f"Parallel execution set to: {'Yes' if parallel_mode else 'No'}")
                
                # If enabling parallelism, also ask about workers
                if parallel_mode:
                    try:
                        new_workers = input(f"Number of parallel workers [{workers}]: ").strip()
                        if new_workers and int(new_workers) > 0:
                            if not hasattr(self.args, "num_workers"):
                                self.args.num_workers = int(new_workers)
                            else:
                                self.args.num_workers = int(new_workers)
                            print(f"Number of parallel workers set to: {new_workers}")
                    except ValueError:
                        print("Invalid value. Using default number of workers.")
        
        elif sub_choice == "3" and current_parallel:
            try:
                new_workers = input(f"Number of parallel workers [{workers}]: ").strip()
                if new_workers and int(new_workers) > 0:
                    if not hasattr(self.args, "num_workers"):
                        self.args.num_workers = int(new_workers)
                    else:
                        self.args.num_workers = int(new_workers)
                    print(f"Number of parallel workers set to: {new_workers}")
            except ValueError:
                print("Invalid value. Please enter a number.")
        
        # Option 4 or invalid option just returns to the settings menu
    
    def _extensions_and_browser_state_settings(self):
        """Configure Chrome extensions and browser state settings."""
        print("\n-- Extensions and Browser State Settings --")
        
        # Extensions directory
        current_extensions_dir = self.env_args.get("extensions_dir", None)
        extensions_status = current_extensions_dir if current_extensions_dir else "No extensions loaded"
        print(f"1. Chrome extensions directory: {extensions_status}")
        
        # Browser user data (profile) directory
        current_user_data_dir = self.env_args.get("golden_user_data_dir", None)
        profile_status = current_user_data_dir if current_user_data_dir else "Using temporary profile"
        print(f"2. Browser user data directory: {profile_status}")
        
        # Viewport size
        current_viewport = self.env_args.get("viewport", {"width": 1280, "height": 720})
        print(f"3. Browser viewport size: {current_viewport['width']}x{current_viewport['height']}")
        
        print("4. Return to settings menu")
        
        sub_choice = input("\nEnter setting to change (1-4): ").strip()
        
        if sub_choice == "1":
            extensions_prompt = "Enter Chrome extensions directory"
            if current_extensions_dir:
                extensions_prompt += f" [{current_extensions_dir}]"
            extensions_prompt += " (empty to disable): "
            
            new_extensions_dir = input(extensions_prompt).strip()
            
            if new_extensions_dir == "":
                # User wants to disable extensions
                self.env_args["extensions_dir"] = None
                if hasattr(self.args, "extensions_dir"):
                    self.args.extensions_dir = None
                print("Chrome extensions disabled")
            elif new_extensions_dir:
                # Verify the directory exists
                if os.path.isdir(new_extensions_dir):
                    self.env_args["extensions_dir"] = new_extensions_dir
                    if hasattr(self.args, "extensions_dir"):
                        self.args.extensions_dir = new_extensions_dir
                    print(f"Chrome extensions directory set to: {new_extensions_dir}")
                else:
                    print(f"Directory {new_extensions_dir} does not exist. Extensions setting unchanged.")
        
        elif sub_choice == "2":
            profile_prompt = "Enter browser user data directory"
            if current_user_data_dir:
                profile_prompt += f" [{current_user_data_dir}]"
            profile_prompt += " (empty to use temporary profile): "
            
            new_user_data_dir = input(profile_prompt).strip()
            
            if new_user_data_dir == "":
                # User wants to use a temporary profile
                self.env_args["golden_user_data_dir"] = None
                if hasattr(self.args, "golden_user_data_dir"):
                    self.args.golden_user_data_dir = None
                print("Using temporary browser profile")
            elif new_user_data_dir:
                # Verify the directory exists
                if os.path.isdir(new_user_data_dir):
                    self.env_args["golden_user_data_dir"] = new_user_data_dir
                    if hasattr(self.args, "golden_user_data_dir"):
                        self.args.golden_user_data_dir = new_user_data_dir
                    print(f"Browser user data directory set to: {new_user_data_dir}")
                else:
                    print(f"Directory {new_user_data_dir} does not exist. User data directory setting unchanged.")
        
        elif sub_choice == "3":
            try:
                new_width = input(f"Enter viewport width [{current_viewport['width']}]: ").strip()
                new_height = input(f"Enter viewport height [{current_viewport['height']}]: ").strip()
                
                width = int(new_width) if new_width else current_viewport['width']
                height = int(new_height) if new_height else current_viewport['height']
                
                if width > 0 and height > 0:
                    self.env_args["viewport"] = {"width": width, "height": height}
                    if hasattr(self.args, "viewport_width"):
                        self.args.viewport_width = width
                    if hasattr(self.args, "viewport_height"):
                        self.args.viewport_height = height
                    print(f"Viewport size set to: {width}x{height}")
                else:
                    print("Invalid dimensions. Viewport size unchanged.")
            except ValueError:
                print("Invalid value. Please enter numbers for width and height.")
    
    def _save_load_menu(self):
        """Menu for saving and loading sessions."""
        print("\n-- Save/Load Session --")
        print("1. Save session")
        print("2. Load session")
        print("3. Return to main menu")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            default_filename = os.path.join(self.args.results_dir, "realbench_session.json")
            filename = input(f"Enter filename [{default_filename}]: ").strip()
            if not filename:
                filename = default_filename
                
            try:
                # Create a deep copy with Path objects converted to strings
                serializable_results = {}
                for task_name, record in self.session_results.items():
                    serializable_record = {}
                    for key, value in record.items():
                        # Convert Path objects to strings
                        if isinstance(value, Path):
                            serializable_record[key] = str(value)
                        else:
                            serializable_record[key] = value
                    serializable_results[task_name] = serializable_record
                
                with open(filename, "w") as f:
                    json.dump(serializable_results, f, indent=2)
                print(f"Saved {len(self.session_results)} results to {filename}")
            except Exception as e:
                print(f"Error saving session: {e}")
                
        elif choice == "2":
            default_filename = os.path.join(self.args.results_dir, "realbench_session.json")
            filename = input(f"Enter filename to load [{default_filename}]: ").strip()
            if not filename:
                filename = default_filename
                
            try:
                with open(filename, "r") as f:
                    self.session_results = json.load(f)
                print(f"Loaded {len(self.session_results)} results from {filename}")
            except Exception as e:
                print(f"Error loading session: {e}")
                
        # For choice 3 or invalid, just return
    
    def _save_session(self):
        """Save the session automatically."""
        try:
            filepath = os.path.join(self.args.results_dir, "realbench_session.json")
            
            # Create a deep copy with Path objects converted to strings
            serializable_results = {}
            for task_name, record in self.session_results.items():
                serializable_record = {}
                for key, value in record.items():
                    # Convert Path objects to strings
                    if isinstance(value, Path):
                        serializable_record[key] = str(value)
                    else:
                        serializable_record[key] = value
                serializable_results[task_name] = serializable_record
            
            with open(filepath, "w") as f:
                json.dump(serializable_results, f, indent=2)
        except Exception as e:
            logger.warning(f"Error auto-saving session: {e}")
            
    def _caching_settings(self):
        """Configure result caching settings."""
        print("\n-- Caching Settings --")
        
        # Get current caching settings with defaults
        use_cache = getattr(self.args, "use_cache", True)
        cache_only = getattr(self.args, "cache_only", False)
        force_refresh = getattr(self.args, "force_refresh", False)
        
        # Show current settings
        print(f"1. Use cached results: {'Yes' if use_cache else 'No'}")
        print(f"2. Cache-only mode (skip tasks without cached results): {'Yes' if cache_only else 'No'}")
        print(f"3. Force refresh (ignore cache, re-run all tasks): {'Yes' if force_refresh else 'No'}")
        print(f"4. View cache status")
        print(f"5. Return to settings menu")
        
        sub_choice = input("\nEnter setting to change (1-5): ").strip()
        
        if sub_choice == "1":
            new_value = input(f"Use cached results? (yes/no) [{'yes' if use_cache else 'no'}]: ").strip().lower()
            if new_value:
                self.args.use_cache = new_value in ["y", "yes", "true", "1"]
                print(f"Use cached results set to: {'Yes' if self.args.use_cache else 'No'}")
        
        elif sub_choice == "2":
            new_value = input(f"Cache-only mode? (yes/no) [{'yes' if cache_only else 'no'}]: ").strip().lower()
            if new_value:
                self.args.cache_only = new_value in ["y", "yes", "true", "1"]
                print(f"Cache-only mode set to: {'Yes' if self.args.cache_only else 'No'}")
        
        elif sub_choice == "3":
            new_value = input(f"Force refresh? (yes/no) [{'yes' if force_refresh else 'no'}]: ").strip().lower()
            if new_value:
                self.args.force_refresh = new_value in ["y", "yes", "true", "1"]
                print(f"Force refresh set to: {'Yes' if self.args.force_refresh else 'No'}")
        
        elif sub_choice == "4":
            # Load and display cache info
            from realeval import find_experiment_dirs, get_experiment_info, get_cache_stats
            
            print("\nScanning experiment directories for cache info...")
            cache_stats = get_cache_stats(self.args.results_dir)
            
            # Get experiment directories
            exp_dirs = find_experiment_dirs(self.args.results_dir)
            
            # Extract experiment info
            experiments = []
            for exp_dir in exp_dirs:
                info = get_experiment_info(exp_dir)
                if info:
                    experiments.append(info)
            
            # Group by cache key (task + config)
            cache_configs = {}
            for exp in experiments:
                cache_key = exp.get("cache_key")
                if not cache_key:
                    continue
                    
                if cache_key not in cache_configs:
                    cache_configs[cache_key] = []
                cache_configs[cache_key].append(exp)
            
            print(f"\nResults directory: {self.args.results_dir}")
            print(f"Total experiment directories found: {len(exp_dirs)}")
            print(f"Total valid experiment results: {len(experiments)}")
            print(f"Unique task configurations: {cache_stats['unique_configs']}")
            
            # Print models in the cache
            if cache_stats['models']:
                print("\nTasks by model:")
                for model, count in sorted(cache_stats['models'].items(), key=lambda x: x[1], reverse=True):
                    print(f"  {model}: {count} tasks")
            
            # Print task types in the cache
            if cache_stats['task_types']:
                print("\nTasks by type:")
                for task_type, count in sorted(cache_stats['task_types'].items(), key=lambda x: x[1], reverse=True):
                    print(f"  {task_type}: {count} tasks")
            
            # Show some specific cache entries with history
            if cache_configs:
                print("\nSample tasks with history:")
                
                # Get up to 5 random keys to display
                import random
                sample_keys = random.sample(list(cache_configs.keys()), min(5, len(cache_configs)))
                
                for key in sample_keys:
                    entries = cache_configs[key]
                    task_name = key.split('_')[0]
                    print(f"\n  {task_name} ({len(entries)} versions):")
                    
                    # Sort by timestamp, newest first
                    sorted_entries = sorted(entries, key=lambda e: e.get("timestamp", 0), reverse=True)
                    
                    # Show the 2 most recent entries
                    for i, entry in enumerate(sorted_entries[:2]):
                        timestamp = entry.get("timestamp", 0)
                        
                        # Convert timestamp to readable date
                        from datetime import datetime
                        date_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                        
                        exp_dir = entry.get("exp_dir", "unknown")
                        reward = entry.get("cum_reward", 0)
                        success = "✓" if reward == 1 else "✗"
                        print(f"    {i+1}. {date_str} - {success} - {exp_dir}")
            
            # Option to browse all cache entries
            print("\nOptions:")
            print("1. View all experiment results")
            print("2. Return to caching settings")
            
            view_choice = input("\nEnter choice (1-2): ").strip()
            
            if view_choice == "1":
                print("\n== All Experiment Results ==")
                
                # Group experiments by task name
                task_experiments = {}
                for exp in experiments:
                    task_name = exp.get("task_name", "unknown")
                    if task_name not in task_experiments:
                        task_experiments[task_name] = []
                    task_experiments[task_name].append(exp)
                
                for task_name, exps in sorted(task_experiments.items()):
                    print(f"\n{task_name} ({len(exps)} versions):")
                    
                    # Sort by timestamp (newest first)
                    sorted_exps = sorted(exps, key=lambda x: x.get("timestamp", 0), reverse=True)
                    
                    for i, exp in enumerate(sorted_exps):
                        timestamp = exp.get("timestamp", 0)
                        from datetime import datetime
                        date_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                        
                        model_name = exp.get("model_name", "unknown")
                        reward = exp.get("cum_reward", 0)
                        success = "✓" if reward == 1 else "✗"
                        exp_dir = exp.get("exp_dir", "unknown")
                        
                        print(f"  {i+1}. {date_str} - {model_name} - {success} - {exp_dir}")
                
                input("\nPress Enter to return to caching settings...")
            # For option 2 or invalid input, just continue
            
        # Option 5 or invalid option just returns to the settings menu
            
    def _quick_actions_menu(self):
        """Menu for quick commonly used actions."""
        print("\n-- Quick Actions --")
        print("1. Run all tasks with AI in parallel mode")
        print("2. Run all tasks for specific task type with AI")
        print("3. Return to main menu")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            self._run_all_tasks_parallel()
        elif choice == "2":
            self._run_tasks_by_type()
        # Option 4 or invalid option returns to main menu
    
    def _run_all_tasks_parallel(self):
        """Run all tasks in parallel with AI agent."""
        print("\n-- Run All Tasks in Parallel --")
        
        # Get model
        default_model = getattr(self.args, "model", "gpt-4o")
        model = input(f"Enter AI model name [{default_model}]: ").strip()
        if not model:
            model = default_model
        
        # Get number of workers
        default_workers = getattr(self.args, "num_workers", 4)
        workers_input = input(f"Enter number of parallel workers [{default_workers}]: ").strip()
        if workers_input and workers_input.isdigit():
            num_workers = int(workers_input)
        else:
            num_workers = default_workers
        
        # Confirm headless mode
        headless_input = input("Run in headless mode (invisible browser)? (yes/no) [yes]: ").strip().lower()
        headless = headless_input != "no" and headless_input != "n"
        
        # Get all tasks
        tasks = get_tasks()
        
        if not tasks:
            print("No tasks found.")
            return
        
        # Confirm execution
        print(f"\nReady to run {len(tasks)} tasks in parallel using {num_workers} workers.")
        print(f"Model: {model}")
        print(f"Headless mode: {'Yes' if headless else 'No'}")
        confirm = input("Proceed? (yes/no): ").strip().lower()
        
        if confirm not in ["y", "yes"]:
            print("Operation cancelled.")
            return
        
        # Update settings
        self.args.model = model
        self.args.num_workers = num_workers
        self.env_args["headless"] = headless

        # Create agent args
        agent_args = create_agent_args("ai", 
            model_name=model,
            chat_mode=False,
            demo_mode="default",
            use_html=self.args.use_html,
            use_axtree=self.args.use_axtree,
            use_screenshot=self.args.use_screenshot,
            system_message_handling="combined" if model == "o1-mini" else "separate"
        ) if model != "operator" else OperatorAgentArgs()
        
        # Run tasks
        print(f"\nRunning {len(tasks)} tasks in parallel with {num_workers} workers...")
        
        try:
            results = run_tasks(
                tasks=tasks,
                agent_args=agent_args,
                env_args_dict=self.env_args,
                results_dir=self.args.results_dir,
                parallel=True,
                num_workers=num_workers,
                use_cache=getattr(self.args, "use_cache", True),
                cache_only=False,
                force_refresh=False
            )
            
            # Store results
            self.session_results.update(results)
            
            # Format and display results
            format_benchmark_results(results)
            
            # Save session after run
            self._save_session()
            
        except Exception as e:
            print(f"Error running tasks: {e}")
    
    def _run_tasks_by_type(self):
        """Run all tasks for a specific task type with AI agent."""
        print("\n-- Run Tasks by Type --")
        
        # Get available task types
        task_types = self.available_task_types
        
        print("Available task types:")
        for i, task_type in enumerate(task_types, 1):
            print(f"{i}. {task_type}")
        
        # Get task type selection
        type_choice = input(f"\nSelect task type (1-{len(task_types)}): ").strip()
        try:
            type_idx = int(type_choice) - 1
            if not (0 <= type_idx < len(task_types)):
                print("Invalid selection.")
                return
        except ValueError:
            print("Invalid input.")
            return
            
        selected_type = task_types[type_idx]
        
        # Get model
        default_model = getattr(self.args, "model", "gpt-4o")
        model = input(f"Enter AI model name [{default_model}]: ").strip()
        if not model:
            model = default_model
        
        # Confirm headless mode
        headless_input = input("Run in headless mode (invisible browser)? (yes/no) [yes]: ").strip().lower()
        headless = headless_input != "no" and headless_input != "n"
        
        # Confirm parallel execution
        parallel_input = input("Run in parallel? (yes/no) [yes]: ").strip().lower()
        parallel = parallel_input != "no" and parallel_input != "n"
        
        if parallel:
            # Get number of workers
            default_workers = getattr(self.args, "num_workers", 4)
            workers_input = input(f"Enter number of parallel workers [{default_workers}]: ").strip()
            if workers_input and workers_input.isdigit():
                num_workers = int(workers_input)
            else:
                num_workers = default_workers
        else:
            num_workers = 1
        
        # Get tasks for the selected type
        tasks = get_tasks(task_type=selected_type)
        
        if not tasks:
            print(f"No tasks found for type '{selected_type}'.")
            return
        
        # Confirm execution
        print(f"\nReady to run {len(tasks)} tasks for type '{selected_type}'.")
        print(f"Model: {model}")
        print(f"Parallel: {'Yes with ' + str(num_workers) + ' workers' if parallel else 'No'}")
        print(f"Headless mode: {'Yes' if headless else 'No'}")
        confirm = input("Proceed? (yes/no): ").strip().lower()
        
        if confirm not in ["y", "yes"]:
            print("Operation cancelled.")
            return
        
        # Update settings
        self.args.model = model
        self.args.num_workers = num_workers
        self.env_args["headless"] = headless
        
        # Create agent args
        agent_args = create_agent_args("ai", 
            model_name=model,
            chat_mode=False,
            demo_mode="default",
            use_html=self.args.use_html,
            use_axtree=self.args.use_axtree,
            use_screenshot=self.args.use_screenshot,
            system_message_handling="combined" if model == "o1-mini" else "separate"
        ) if model != "operator" else OperatorAgentArgs()
        
        # Run tasks
        print(f"\nRunning {len(tasks)} tasks for '{selected_type}'...")
        
        try:
            results = run_tasks(
                tasks=tasks,
                agent_args=agent_args,
                env_args_dict=self.env_args,
                results_dir=self.args.results_dir,
                parallel=parallel,
                num_workers=num_workers,
                use_cache=getattr(self.args, "use_cache", True),
                cache_only=False,
                force_refresh=False
            )
            
            # Store results
            self.session_results.update(results)
            
            # Format and display results
            format_benchmark_results(results)
            
            # Save session after run
            self._save_session()
            
        except Exception as e:
            print(f"Error running tasks: {e}")
    
    def _continue_benchmark(self):
        """Run any tasks missing from the cache."""
        print("\n-- Continue Benchmark (Run Uncached Tasks) --")
        
        # Get model
        default_model = getattr(self.args, "model", "gpt-4o")
        model = input(f"Enter AI model name [{default_model}]: ").strip()
        if not model:
            model = default_model
        
        # Get number of workers
        default_workers = getattr(self.args, "num_workers", 4)
        workers_input = input(f"Enter number of parallel workers [{default_workers}]: ").strip()
        if workers_input and workers_input.isdigit():
            num_workers = int(workers_input)
        else:
            num_workers = default_workers
        
        # Confirm headless mode
        headless_input = input("Run in headless mode (invisible browser)? (yes/no) [yes]: ").strip().lower()
        headless = headless_input != "no" and headless_input != "n"
        
        # Determine which tasks are not cached
        from realeval import create_cache_key, find_cached_result
        
        # Get all tasks
        all_tasks = get_tasks()
        
        # Create temporary agent args to check against cache
        temp_agent_args = create_agent_args("ai", 
            model_name=model,
            chat_mode=False,
            demo_mode="default",
            use_html=self.args.use_html,
            use_axtree=self.args.use_axtree,
            use_screenshot=self.args.use_screenshot,
            system_message_handling="combined" if model == "o1-mini" else "separate"
        ) if model != "operator" else OperatorAgentArgs()
        
        # Check which tasks are cached
        uncached_tasks = []
        
        print("Scanning for uncached tasks...")
        for task in all_tasks:
            cached_result = find_cached_result(task, temp_agent_args, self.env_args, self.args.results_dir)
            if not cached_result:
                uncached_tasks.append(task)
        
        if not uncached_tasks:
            print("All tasks are already cached. Nothing to do.")
            return
        
        # Confirm execution
        print(f"\nFound {len(uncached_tasks)} uncached tasks out of {len(all_tasks)} total tasks.")
        print(f"Model: {model}")
        print(f"Parallel: Yes with {num_workers} workers")
        print(f"Headless mode: {'Yes' if headless else 'No'}")
        confirm = input("Proceed to run uncached tasks? (yes/no): ").strip().lower()
        
        if confirm not in ["y", "yes"]:
            print("Operation cancelled.")
            return
        
        # Update settings
        self.args.model = model
        self.args.num_workers = num_workers
        self.env_args["headless"] = headless
        
        # Create agent args
        agent_args = create_agent_args("ai", 
            model_name=model,
            chat_mode=False,
            demo_mode="default",
            use_html=self.args.use_html,
            use_axtree=self.args.use_axtree,
            use_screenshot=self.args.use_screenshot,
            system_message_handling="combined" if model == "o1-mini" else "separate"
        ) if model != "operator" else OperatorAgentArgs()
        
        # Run tasks
        print(f"\nRunning {len(uncached_tasks)} uncached tasks...")
        
        try:
            results = run_tasks(
                tasks=uncached_tasks,
                agent_args=agent_args,
                env_args_dict=self.env_args,
                results_dir=self.args.results_dir,
                parallel=True,
                num_workers=num_workers,
                use_cache=True,
                cache_only=False,
                force_refresh=False
            )
            
            # Store results
            self.session_results.update(results)
            
            # Format and display results
            format_benchmark_results(results)
            
            # Save session after run
            self._save_session()
            
        except Exception as e:
            print(f"Error running tasks: {e}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Interactive guided RealBench evaluation")
    
    # Environment options
    env_group = parser.add_argument_group("Environment Configuration")
    env_group.add_argument("--golden-user-data-dir", type=str, help="Path to a user data directory to use as a golden profile")
    env_group.add_argument("--extensions-dir", type=str, help="Path to a directory containing Chrome extensions to load")
    env_group.add_argument("--viewport-width", type=int, default=1280, help="Width of the browser viewport")
    env_group.add_argument("--viewport-height", type=int, default=720, help="Height of the browser viewport")
    env_group.add_argument("--max-steps", type=int, default=25, help="Maximum number of steps per task")
    
    # Browser visibility options
    headless_group = env_group.add_mutually_exclusive_group()
    headless_group.add_argument("--headless", dest="headless", action="store_true", help="Run in headless mode (hide browser)")
    headless_group.add_argument("--no-headless", dest="headless", action="store_false", help="Show browser (disable headless mode)")
    parser.set_defaults(headless=False)  # Default to showing browser for interactive sessions
    
    # AI agent options
    ai_group = parser.add_argument_group("AI Agent Configuration")
    ai_group.add_argument("--model", default="gpt-4o", help="Model to use for AI agent")
    ai_group.add_argument("--use-html", action="store_true", help="Include HTML in observations")
    ai_group.add_argument("--use-axtree", action="store_true", default=True, help="Include accessibility tree in observations")
    ai_group.add_argument("--use-screenshot", action="store_true", default=True, help="Include screenshots in observations")
    
    # Parallelization options
    parallel_group = parser.add_argument_group("Parallelization Settings")
    parallel_group.add_argument("--parallel", action="store_true", help="Run tasks in parallel")
    parallel_group.add_argument("--num-workers", type=int, default=4, help="Number of parallel workers to use (only used with --parallel)")
    
    # Caching options
    cache_group = parser.add_argument_group("Caching Settings")
    cache_group.add_argument("--no-cache", dest="use_cache", action="store_false", 
                            help="Disable result caching (will always run all tasks)")
    cache_group.add_argument("--cache-only", action="store_true", 
                            help="Only use cached results, don't run missing tasks")
    cache_group.add_argument("--force-refresh", action="store_true", 
                            help="Force re-running tasks even if cached results exist")
    parser.set_defaults(use_cache=True)
    
    # Special parameters
    special_group = parser.add_argument_group("Special Parameters")
    special_group.add_argument("--start-url", default="https://www.google.com", help="Starting URL for openended task")
    special_group.add_argument("--run-id", type=str, help="Specific run ID to use for tasks (needed for leaderboard submissions)")
    
    # Output options
    output_group = parser.add_argument_group("Output Settings")
    output_group.add_argument("--results-dir", default="./results", help="Directory to store results")
    
    return parser.parse_args()

def main():
    """Main entry point for the guided interactive session."""
    args = parse_args()
    
    # Start the guided session
    session = GuidedRealBenchSession(args)
    session.start()

if __name__ == "__main__":
    main()