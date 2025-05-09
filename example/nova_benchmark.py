
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import requests  # pip install requests
from nova_act import NovaAct
from agisdk.REAL.tasks import all_tasks as tasks

###############################################################################
# Realevals helpers
###############################################################################

# Register a run and return its *run_id*
def get_run_id(api_key: str, model_name: str, run_name: str) -> str:
    url = (
            "https://www.realevals.xyz/api/runKey?"
            + urllib.parse.urlencode(
                {
                    "api_key": api_key,
                    "model_name": model_name,
                    "run_name": run_name,
                }
            )
        )
    response = requests.get(url, timeout=30)
    print(f"API Response: {response.json()}")
    return response.json()["newRunId"]

# Get the results of a run by display name
def get_run_results(api_key: str, display_name: str) -> dict:
    url = (
            "https://www.realevals.xyz/api/getRunTask?"
            + urllib.parse.urlencode(
                {
                    "api_key": api_key,
                    "display_name": display_name,
                }
            )
        )
    try:
        response = requests.get(url, timeout=30)
        data = response.json()
        
        # Print overall data
        print(f"\n===== Results for {display_name} =====")
        print(f"Run ID: {data.get('run_id', 'N/A')}")
        print(f"Model ID: {data.get('model_id', 'N/A')}")
        print(f"Success Rate: {data.get('success_rate', 0)}%")
        print(f"Total Runs: {data.get('total_runs', 0)}")
        print(f"Created At: {data.get('created_at', 'N/A')}")
        
        # Print detailed run data
        if 'runs' in data and data['runs']:
            print("\n----- Individual Runs -----")
            for i, run in enumerate(data['runs']):
                print(f"\nRun #{i+1}: {run.get('task_id', 'N/A')}")
                print(f"  Retrieved Answer: {run.get('retrieved_answer', 'N/A')[:50]}...")
                print(f"  Evals Passed: {run.get('evals_passed', 'N/A')}")
                print(f"  Evals Failed: {run.get('evals_failed', 'N/A')}")
                print(f"  Points: {run.get('points', 'N/A')}")
                print(f"  Accuracy: {run.get('accuracy', 'N/A')}")
                print(f"  Completed At: {run.get('completed_at', 'N/A')}")
                print(f"  Final State: {run.get('final_state', 'N/A')}")
        
        return data
    except Exception as e:
        print(f"Error fetching results: {e}")
        return {"error": str(e)}

###############################################################################
# NovaAct execution
###############################################################################

def run_task(task: dict, run_id: str, headless: bool) -> dict:
    """Execute *task* with NovaAct and submit the answer."""
    t0 = time.time()
    tid, goal, base = task["id"], task["goal"], task["website"]["url"]
    cfg = f"{base}/config?run_id={run_id}&task_id={tid}"
    
    result = {
        "id": tid,
        "task_id": tid,
        "start_time": datetime.now().isoformat(),
        "ok": False,
        "success": False,
        "error": None,
        "response": None,
        "elapsed_time": 0,
        "t": 0
    }

    try:
        print(f"Starting task: {tid} - {goal[:50]}...")
        with NovaAct(starting_page=cfg, headless=headless) as bot:
            bot.go_to_url(base)
            start_act_time = time.time()
            nova_result = bot.act(goal)
            act_elapsed = time.time() - start_act_time
            
            print(f"Task {tid} action completed in {act_elapsed:.2f} seconds")
            answer = nova_result.response or ""
            result["response"] = answer
            
            # Submit the response
            bot.go_to_url(
                f"{base}/submit?retrieved_answer={urllib.parse.quote(answer)}"
            )
            
            result["ok"] = True
            result["success"] = True
    except Exception as exc:
        result["error"] = str(exc)
        print(f"Error on task {tid}: {exc}")
    
    elapsed = time.time() - t0
    result["elapsed_time"] = elapsed
    result["t"] = elapsed
    result["end_time"] = datetime.now().isoformat()
    
    print(f"Completed task {tid} in {elapsed:.2f} seconds. Success: {result['success']}")
    return result

###############################################################################
# CLI entrypoint
###############################################################################

def main() -> None:
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"NovaAct_{ts}"
    run_name = f"NovaAct_{ts}"
    api_key = "b0c2f93d0c461eed5b99b51ed6e934baa600ba0907185edd93c949ab20f34d21"
    run_id = get_run_id(api_key, model_name, run_name)
    
    p = argparse.ArgumentParser("Tiny NovaAct benchmark")
    p.add_argument("--api-key", default=api_key)
    p.add_argument("--run-name", default=run_name)
    p.add_argument("--workers", type=int, default=1)  # Reduced to 1 worker
    p.add_argument("--filter", default="omnizon-1")  # Default to just one task
    p.add_argument("--no-headless", action="store_true")
    p.add_argument("--run-id", default=run_id)
    args = p.parse_args()
    
    selected = [t for t in tasks if args.filter in t["id"]] if args.filter else tasks
    print(f"{len(selected)} tasks → {args.workers} workers")

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_task, t, run_id, not args.no_headless) for t in selected]
        for f in as_completed(futures):
            results.append(f.result())

    # Calculate summary statistics
    successful_tasks = [r for r in results if r["ok"]]
    success_rate = len(successful_tasks) / len(results) if results else 0
    avg_time = sum(r["elapsed_time"] for r in results) / len(results) if results else 0
    
    print(f"✓ {len(successful_tasks)}/{len(results)} tasks succeeded ({success_rate:.2%})")
    print(f"Average time per task: {avg_time:.2f} seconds")

    out = {
        "timestamp": ts,
        "run_id": run_id,
        "model_name": model_name,
        "run_name": run_name,
        "total_tasks": len(results),
        "successful_tasks": len(successful_tasks),
        "success_rate": success_rate,
        "average_time": avg_time,
        "results": results,
    }
    # Print summary instead of saving to file
    print("\nRun Summary:")
    print(f"Timestamp: {ts}")
    print(f"Run ID: {run_id}")
    print(f"Tasks: {len(results)}")
    print(f"Success: {len(successful_tasks)}")
    print(f"Success Rate: {success_rate:.2%}")
    print(f"Avg Time: {avg_time:.2f} seconds")


if __name__ == "__main__":
    main()