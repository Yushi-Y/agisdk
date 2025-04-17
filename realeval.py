#!/usr/bin/env python3
"""
Core library for RealBench task evaluation and benchmarking.
Provides shared functionality for both AI and human evaluation.
"""

import os
import glob
import random
import json
import time
import dataclasses
import logging
import tempfile
import shutil
from typing import List, Dict, Optional, Any, Union, Literal, Tuple, Type
from pathlib import Path
from statistics import mean, median, stdev
import multiprocessing as mp
from functools import partial

# Ensure tempfile is accessible in all scopes, including those that use
# multiprocessing which might not properly inherit imported modules
import tempfile as tempfile_module
tempfile = tempfile_module

from agisdk.real.browsergym.experiments import Agent, AbstractAgentArgs, EnvArgs, ExpArgs, get_exp_result

# Worker initialization function for multiprocessing
def init_worker():
    global tempfile
    import tempfile

logger = logging.getLogger(__name__)

# Agent factory and registry
AGENT_REGISTRY = {}

def register_agent(agent_type: str, agent_args_class: Type[AbstractAgentArgs]):
    """Register an agent type with its arguments class."""
    AGENT_REGISTRY[agent_type] = agent_args_class

def create_agent_args(agent_type: str, **kwargs) -> AbstractAgentArgs:
    """Create agent arguments for the specified agent type."""
    if agent_type not in AGENT_REGISTRY:
        raise ValueError(f"Unknown agent type: {agent_type}. Available types: {list(AGENT_REGISTRY.keys())}")
    return AGENT_REGISTRY[agent_type](**kwargs)

def get_available_task_types() -> List[str]:
    """Get all available task types from the tasks directory."""
    tasks_dir = Path(__file__).parent / "src" / "agisdk" / "real" / "browsergym" / "webclones" / "tasks"
    
    # Get all JSON files in the main tasks directory (excluding the alt subdirectory)
    json_files = [f for f in glob.glob(f"{tasks_dir}/*.json")]
    
    # Extract task types by taking the prefix before the first dash followed by a number
    task_types = set()
    for f in json_files:
        basename = os.path.basename(f)
        # Match the pattern up to the first dash followed by a number
        parts = basename.split('-')
        # Find the index of the first part that is a number
        for i, part in enumerate(parts[1:], 1):
            if part and part[0].isdigit():
                task_types.add('-'.join(parts[:i]))
                break
    return sorted(task_types)

def create_cache_key(task_name: str, agent_args: AbstractAgentArgs, env_args_dict: Dict[str, Any]) -> str:
    """
    Create a unique cache key for a task-agent-env combination.
    
    Args:
        task_name: Name of the task
        agent_args: Arguments for the agent
        env_args_dict: Dictionary of arguments for the environment
        
    Returns:
        A string key for the cache
    """
    # Extract core agent info for the cache key
    agent_model = getattr(agent_args, "model_name", "unknown")
    agent_type = agent_args.agent_name if hasattr(agent_args, "agent_name") else type(agent_args).__name__
    
    # Extract core environment settings
    max_steps = env_args_dict.get("max_steps", "default")
    
    # Create a reproducible cache key
    cache_key = f"{task_name}_{agent_type}_{agent_model}_{max_steps}"
    
    return cache_key

def find_experiment_dirs(results_dir: str) -> List[Path]:
    """
    Find all experiment directories in the results directory.
    
    Args:
        results_dir: Directory containing experiment results
        
    Returns:
        List of experiment directory paths
    """
    results_path = Path(results_dir)
    
    # Experiment directories are identified by the presence of summary_info.json
    exp_dirs = []
    
    # Walk through all directories in results_dir
    for root, dirs, files in os.walk(results_dir):
        root_path = Path(root)
        
        # Check if this directory has the required files to be an experiment directory
        if "summary_info.json" in files:
            exp_dirs.append(root_path)
    
    return exp_dirs

def get_experiment_info(exp_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Extract information about an experiment from its directory.
    
    Args:
        exp_dir: Path to experiment directory
        
    Returns:
        Dictionary with experiment information, or None if info can't be extracted
    """
    try:
        # Get modification time as timestamp
        timestamp = os.path.getmtime(exp_dir / "summary_info.json")
        
        # Load summary info
        with open(exp_dir / "summary_info.json", "r") as f:
            summary_info = json.load(f)
        
        # Extract relevant information directly from summary_info.json
        task_name = summary_info.get("task_name")
        agent_type = summary_info.get("agent_type")
        model_name = summary_info.get("model_name", "unknown")
        max_steps = summary_info.get("max_steps")
        
        # Create complete info dict
        info = {
            "task_name": task_name,
            "agent_type": agent_type,
            "model_name": model_name,
            "max_steps": max_steps,
            "timestamp": timestamp,
            "exp_dir": str(exp_dir),
            "cache_key": f"{task_name}_{agent_type}_{model_name}_{max_steps}",
        }
        
        # Add any other summary information
        info.update(summary_info)
        
        return info
    except Exception as e:
        print(f"Warning: Failed to extract info from {exp_dir}: {e}")
        return None


def get_tasks(
    task_type: Optional[str] = None,
    task_id: Optional[int] = None,
    sample_size: Optional[int] = None,
    random_seed: int = 42,
    include_impossible: bool = False
) -> List[str]:
    """
    Get tasks based on filtering criteria.
    
    Args:
        task_type: Filter tasks by type (e.g., 'omnizon', 'dashdish')
        task_id: Run a specific task ID for the given task type
        sample_size: Number of tasks to sample (after filtering by task_type)
        random_seed: Seed for random sampling
        include_impossible: Whether to include tasks marked as impossible
        
    Returns:
        List of task names formatted as 'webclones.{task_type}-{task_id}'
    """
    tasks_dir = Path(__file__).parent / "src" / "agisdk" / "real" / "browsergym" / "webclones" / "tasks"
    
    # Get all JSON files in the main tasks directory (excluding the alt subdirectory)
    json_files = [f for f in glob.glob(f"{tasks_dir}/*.json") if "/alt/" not in f]
    
    filtered_tasks = []
    for f in json_files:
        # If including all tasks, add all files
        if include_impossible:
            filtered_tasks.append(f)
            continue
            
        # Otherwise filter out impossible tasks
        with open(f, 'r') as file:
            try:
                task_data = json.load(file)
                # Only include tasks where "possible" is not explicitly set to false
                if task_data.get('possible', True):
                    filtered_tasks.append(f)
            except json.JSONDecodeError:
                # Skip files with invalid JSON
                continue
    
    # Extract task names without extension
    task_names = [os.path.basename(f).replace('.json', '') for f in filtered_tasks]
    
    # Filter by task type if specified
    if task_type:
        task_names = [t for t in task_names if t.startswith(f"{task_type}-") or 
                      (('-' in t) and t.split('-')[:-1] == task_type.split('-'))]
    
    # Filter by specific task ID if specified
    if task_type and task_id is not None:
        specific_task = f"{task_type}-{task_id}"
        if specific_task in task_names:
            return [f"webclones.{specific_task}"]
        else:
            raise ValueError(f"Task {specific_task} not found")
    
    # Sample tasks if requested
    if sample_size is not None and sample_size > 0:
        if sample_size >= len(task_names):
            # If sample size is larger than available tasks, use all tasks
            pass
        else:
            random.seed(random_seed)
            task_names = random.sample(task_names, sample_size)
    
    # Format task names for browsergym
    return [f"webclones.{name}" for name in sorted(task_names)]

def find_cached_result(task_name: str, agent_args: AbstractAgentArgs, env_args_dict: Dict[str, Any], 
                      results_dir: str) -> Optional[Dict[str, Any]]:
    """
    Find a cached result for the given task and agent configuration.
    
    Args:
        task_name: Name of the task
        agent_args: Arguments for the agent
        env_args_dict: Dictionary of arguments for the environment
        results_dir: Directory containing experiment results
        
    Returns:
        The cached result or None if not found or if the result contains errors
    """
    # Create cache key
    cache_key = create_cache_key(task_name, agent_args, env_args_dict)
    
    # Find all experiment directories
    exp_dirs = find_experiment_dirs(results_dir)
    
    # Filter by cache key
    matching_exps = []
    
    for exp_dir in exp_dirs:
        # Extract experiment info
        info = get_experiment_info(exp_dir)
        
        # If we can't extract info, skip this directory
        if info is None:
            continue
        
        # Check if this experiment matches our cache key
        if info.get("cache_key") == cache_key:
            matching_exps.append(info)
    
    # If no matching experiments, return None
    if not matching_exps:
        return None
    
    # Sort by timestamp (newest first)
    matching_exps.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
    
    # Get the most recent experiment
    result = matching_exps[0]
    
    # Check if the result has errors
    has_error = (result.get("err_msg") is not None or 
                result.get("stack_trace") is not None)
    
    # Always skip results with errors
    if has_error:
        print(f"Skipping cached result for {task_name} due to errors, will rerun")
        return None
        
    return result

def get_cache_stats(results_dir: str) -> Dict[str, Any]:
    """
    Get statistics about the cache by scanning experiment directories.
    
    Args:
        results_dir: Directory containing experiment results
        
    Returns:
        Dictionary with cache statistics including total experiments, unique configs,
        models used, and task types
    """
    # Find all experiment directories
    exp_dirs = find_experiment_dirs(results_dir)
    
    # Track statistics
    models = {}
    task_types = {}
    cache_keys = set()
    valid_experiments = 0
    
    # Process each directory
    for exp_dir in exp_dirs:
        # Get experiment info from summary_info.json
        info = get_experiment_info(exp_dir)
        if not info:
            continue
            
        valid_experiments += 1
        
        # Extract relevant fields
        task_name = info.get("task_name", "")
        model_name = info.get("model_name", "unknown")
        cache_key = info.get("cache_key", "")
        
        # Add to cache keys set for unique config count
        if cache_key:
            cache_keys.add(cache_key)
        
        # Add to models count
        if model_name:
            models[model_name] = models.get(model_name, 0) + 1
        
        # Extract task type from task name (e.g., "webclones.omnizon-1" -> "omnizon")
        if '.' in task_name:
            parts = task_name.split('.')[1].split('-')
            # Handle special cases like "fly-unified" where task type has a hyphen
            for i, part in enumerate(parts[1:], 1):
                if part and part[0].isdigit():
                    task_type = '-'.join(parts[:i])
                    break
            else:
                # Fallback if no numeric part is found
                task_type = parts[0]
                
            task_types[task_type] = task_types.get(task_type, 0) + 1
    
    # Compile statistics
    stats = {
        "total_experiments": valid_experiments,
        "unique_configs": len(cache_keys),
        "models": models,
        "task_types": task_types,
    }
    
    return stats

def run_single_task(
    task_name: str,
    agent_args: AbstractAgentArgs,
    env_args_dict: Dict[str, Any],
    results_dir: str,
    continue_previous: bool = False,
    use_cache: bool = True,
    run_uuid: Optional[str] = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Run a single task with the provided agent and environment configuration.
    
    Args:
        task_name: Name of the task to run
        agent_args: Arguments for the agent
        env_args_dict: Dictionary of arguments for the environment
        results_dir: Directory to store results
        continue_previous: Whether to try to continue from a previous run
        use_cache: Whether to update the cache with results
        run_uuid: Optional UUID for tracking this run batch
        
    Returns:
        Tuple of (task_name, results_dict)
    """
    print(f"Running task: {task_name}")
    
    # Set task name in env args
    env_args_dict["task_name"] = task_name
    
    # Create EnvArgs from dictionary
    env_args = EnvArgs(**env_args_dict)
    
    # Set up experiment
    exp_args = ExpArgs(
        env_args=env_args,
        agent_args=agent_args
    )
    
    # Start timing
    start_time = time.time()
    
    # Run experiment
    exp_args.prepare(results_dir)
    
    # Add essential metadata to summary_info.json before running the experiment
    # This ensures the cache has what it needs even if there's a crash
    summary_info_path = Path(exp_args.exp_dir) / "summary_info.json"
    
    # Extract metadata for cache key
    agent_type = agent_args.agent_name if hasattr(agent_args, "agent_name") else type(agent_args).__name__
    model_name = getattr(agent_args, "model_name", "unknown")
    max_steps = env_args.max_steps
    
    # Create initial summary info with metadata
    initial_summary = {
        "task_name": task_name,
        "agent_type": agent_type,
        "model_name": model_name,
        "max_steps": max_steps,
        "cache_key": f"{task_name}_{agent_type}_{model_name}_{max_steps}",
        "experiment_status": "started",
        "run_uuid": run_uuid,  # Add the run UUID for tracking
    }
    
    # Write initial summary info
    with open(summary_info_path, "w") as f:
        json.dump(initial_summary, f, indent=4)
    
    # Run the experiment
    exp_args.run()
    
    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Get results
    exp_result = get_exp_result(exp_args.exp_dir)
    exp_record = exp_result.get_exp_record()
    
    # Add timing information to the record
    exp_record['elapsed_time'] = elapsed_time
    
    # Add experiment directory to the record
    exp_record['exp_dir'] = str(exp_args.exp_dir)
    
    # Print current task result
    print(f"Task: {task_name}")
    print(f"  Reward: {exp_record.get('cum_reward', 0)}")
    success = exp_record.get('cum_reward', 0) == 1
    print(f"  Success: {success}")
    print(f"  Time: {elapsed_time:.2f} seconds")
    
    return task_name, exp_record

def run_tasks(
    tasks: List[str],
    agent_args: AbstractAgentArgs,
    env_args_dict: Dict[str, Any],
    results_dir: str = "./results",
    parallel: bool = False,
    num_workers: int = 5,
    continue_previous: bool = False,
    use_cache: bool = True,
    cache_only: bool = False,
    force_refresh: bool = False
) -> Dict[str, Any]:
    """
    Run tasks with the provided agent and environment configuration.
    
    Args:
        tasks: List of task names to run
        agent_args: Arguments for the agent
        env_args_dict: Dictionary of arguments for the environment
        results_dir: Directory to store results
        parallel: Whether to run tasks in parallel
        num_workers: Number of workers to use for parallel execution
        continue_previous: Whether to try to continue from previous runs
        use_cache: Whether to use and update the cache
        cache_only: Whether to only use cached results without running missing tasks
        force_refresh: Whether to force re-running tasks even if cached results exist
        
    Returns:
        Dictionary of results indexed by task name
    """
    # Generate a unique run ID for this batch
    import uuid
    from datetime import datetime
    run_uuid = str(uuid.uuid4())
    run_timestamp = datetime.now().isoformat()
    print(f"Running multiple tasks with ID: {run_uuid}")
    # Store run metadata for tracking
    run_metadata = {
        "run_uuid": run_uuid,
        "run_timestamp": run_timestamp,
        "agent_type": agent_args.agent_name if hasattr(agent_args, "agent_name") else type(agent_args).__name__,
        "model_name": getattr(agent_args, "model_name", "unknown"),
        "total_tasks": len(tasks)
    }
    
    print(f"Starting run with ID: {run_uuid}")
    
    # Initialize results dictionary
    results = {}
    
    # Determine which tasks need to be run
    tasks_to_run = []
    
    if use_cache and not force_refresh:
        for task_name in tasks:
            # Try to find a cached result
            cached_result = find_cached_result(task_name, agent_args, env_args_dict, results_dir)
            
            if cached_result:
                # Use cached result
                print(f"Using cached result for {task_name} from {cached_result.get('exp_dir', 'unknown')}")
                results[task_name] = cached_result
            elif not cache_only:
                # Need to run this task
                tasks_to_run.append(task_name)
            else:
                print(f"No cached result for {task_name} (skipping in cache-only mode)")
    else:
        # Force refresh or no cache - run all tasks
        if not cache_only:
            tasks_to_run = tasks
    
    # Run tasks if needed
    if tasks_to_run:
        print(f"Running {len(tasks_to_run)} tasks...")
        
        if parallel:
            # Create a partial function with the fixed arguments
            run_task_partial = partial(
                run_single_task,
                agent_args=agent_args,
                env_args_dict=env_args_dict,
                results_dir=results_dir,
                continue_previous=continue_previous,
                use_cache=use_cache,
                run_uuid=run_uuid
            )
            
            # Run tasks in parallel using multiprocessing
            with mp.Pool(processes=num_workers, initializer=init_worker) as pool:
                new_results = dict(pool.map(run_task_partial, tasks_to_run))
                
                # Merge with cached results
                results.update(new_results)
        else:
            # Run tasks sequentially
            for task_name in tasks_to_run:
                task_name, exp_record = run_single_task(
                    task_name=task_name,
                    agent_args=agent_args,
                    env_args_dict=env_args_dict,
                    results_dir=results_dir,
                    continue_previous=continue_previous,
                    use_cache=use_cache,
                    run_uuid=run_uuid
                )
                results[task_name] = exp_record
    
    # Gather statistics for this run using the run_uuid
    cache_hits = len(tasks) - len(tasks_to_run)
    
    # Find all experiments with this run_uuid to count errors
    exps_with_errors = 0
    total_exps = 0
    
    for exp_dir in find_experiment_dirs(results_dir):
        # Extract experiment info
        info = get_experiment_info(exp_dir)
        
        # If we can't extract info, skip this directory
        if info is None:
            continue
            
        # Count only experiments from this run
        if info.get("run_uuid") == run_uuid:
            total_exps += 1
            
            # Check for errors
            if info.get("err_msg") is not None or info.get("stack_trace") is not None:
                exps_with_errors += 1
    
    # Print statistics
    print("\nRun Statistics:")
    print(f"  Run UUID: {run_uuid}")
    print(f"  Total tasks: {len(tasks)}")
    print(f"  From cache: {cache_hits}")
    print(f"  Newly executed: {len(tasks_to_run)}")
    print(f"  Tasks with errors: {exps_with_errors} of {total_exps} ({exps_with_errors/total_exps*100 if total_exps > 0 else 0:.1f}%)")
    
    return results

def format_benchmark_results(results: Dict[str, Any]) -> None:
    """Format and print aggregated benchmark results."""
    if not results:
        print("No results to display.")
        return
    
    # Calculate aggregate score
    success_count = sum(1 for _, r in results.items() if r.get('cum_reward', 0) == 1)
    success_rate = success_count / len(results) * 100
    
    # Collect timing statistics
    all_times = [r.get('elapsed_time', 0) for _, r in results.items()]
    successful_times = [r.get('elapsed_time', 0) for _, r in results.items() if r.get('cum_reward', 0) == 1]
    
    print("\n===== BENCHMARK RESULTS =====")
    print(f"Tasks completed successfully: {success_count}/{len(results)}")
    print(f"Success rate: {success_rate:.2f}%")
    
    # Print timing statistics
    if all_times:
        print("\nTiming Statistics (All Tasks):")
        print(f"  Average time: {mean(all_times):.2f} seconds")
        print(f"  Median time: {median(all_times):.2f} seconds")
        print(f"  Min time: {min(all_times):.2f} seconds")
        print(f"  Max time: {max(all_times):.2f} seconds")
        if len(all_times) > 1:
            print(f"  Std deviation: {stdev(all_times):.2f} seconds")
    
    if successful_times:
        print("\nTiming Statistics (Successful Tasks Only):")
        print(f"  Average time: {mean(successful_times):.2f} seconds")
        print(f"  Median time: {median(successful_times):.2f} seconds")
        print(f"  Min time: {min(successful_times):.2f} seconds")
        print(f"  Max time: {max(successful_times):.2f} seconds")
        if len(successful_times) > 1:
            print(f"  Std deviation: {stdev(successful_times):.2f} seconds")
    
    # Group results by task type
    task_type_results = {}
    for task_name, record in results.items():
        # Extract task type (e.g., "omnizon" from "webclones.omnizon-1" or "fly-unified" from "webclones.fly-unified-1")
        task_full_name = task_name.split('.')[1]
        parts = task_full_name.split('-')
        
        # Find where the numeric part starts
        for i, part in enumerate(parts[1:], 1):
            if part and part[0].isdigit():
                task_type = '-'.join(parts[:i])
                break
        else:
            # Fallback if no numeric part is found
            task_type = parts[0]
        
        if task_type not in task_type_results:
            task_type_results[task_type] = {
                'total': 0,
                'success': 0,
                'times': []
            }
        
        task_type_results[task_type]['total'] += 1
        task_type_results[task_type]['times'].append(record.get('elapsed_time', 0))
        if record.get('cum_reward', 0) == 1:
            task_type_results[task_type]['success'] += 1
    
    # Print results by task type
    print("\nResults by task type:")
    for task_type, stats in sorted(task_type_results.items()):
        success_rate = (stats['success'] / stats['total']) * 100
        avg_time = mean(stats['times']) if stats['times'] else 0
        print(f"  {task_type}: {stats['success']}/{stats['total']} ({success_rate:.2f}%) - Avg time: {avg_time:.2f}s")

def format_human_results(results: Dict[str, Any]) -> None:
    """Format and print results for human evaluation."""
    if not results:
        print("No results to display.")
        return
    
    print("\n===== HUMAN EVALUATION RESULTS =====")
    
    for task_name, record in results.items():
        print(f"\nTask: {task_name}")
        print(f"  Success: {'Yes' if record.get('cum_reward', 0) == 1 else 'No'}")
        print(f"  Time taken: {record.get('elapsed_time', 0):.2f} seconds")
        
        # Print any custom metrics if available
        for key, value in record.items():
            if key.startswith('custom_') and value is not None:
                print(f"  {key.replace('custom_', '').replace('_', ' ').title()}: {value}")
    
    # Summary statistics
    success_count = sum(1 for _, r in results.items() if r.get('cum_reward', 0) == 1)
    if len(results) > 1:
        print(f"\nSummary: {success_count}/{len(results)} tasks completed successfully")