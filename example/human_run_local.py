import argparse
import json
import os
import glob
from nova_act import NovaAct
from agisdk.REAL.browsergym.webclones.evaluate import WebCloneEvaluator
from agisdk.REAL.browsergym.webclones.task_config import TaskConfig
from datetime import datetime

# Task loading functions
def load_local_tasks(task_filter=None):
    local_tasks = []
    task_files = glob.glob(os.path.join(os.path.dirname(__file__), "tasks/*.json"))
    
    print(f"Found {len(task_files)} task files")
    
    for task_file in task_files:
        with open(task_file, 'r') as f:
            task = json.load(f)
            if task_filter is None or task_filter == "" or task["id"] == task_filter:
                local_tasks.append(task)
    
    return local_tasks

# NovaAct execution
def run_task(task: dict, run_id: str, headless: bool = True) -> dict:
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

    print(f"Starting task: {tid} - {goal[:200]}...")
    try:
        with NovaAct(starting_page=cfg, headless=headless) as bot:
            bot.go_to_url(base)
            try:
                result["response"] = input("Press Enter when ready to finish:")
                
                # Finish the task
                bot.go_to_url(f"{base}/finish")
                    
                pre_element = bot.page.wait_for_selector("pre")
                if pre_element:
                    env_state = json.loads(pre_element.inner_text())
                
                # import code; code.interact(local=locals())
                
                config_json = TaskConfig(os.path.join(os.path.dirname(__file__), f"tasks/{tid}"), is_path=True)
                evaluator = WebCloneEvaluator(task_config=config_json)
                reward, done, message, info = evaluator.evaluate(env_state=env_state, model_response=result["response"])
                print(f"Evaluation result: {message}, Reward: {reward}")
                
                result["ok"] = True
                result["success"] = done
            except Exception as e:
                print(f"Error during act: {e}")
                result["error"] = str(e)
    except Exception as e:
        print(f"Error setting up NovaAct: {e}")
        result["error"] = str(e)
            
    return result


def main():
    p = argparse.ArgumentParser("NovaAct Local Tasks")
    p.add_argument("--filter", default="", help="Task ID filter (empty for all tasks)")
    p.add_argument("--run-id", default="b0c2f93d0c461eed5b99b51ed6e934baa600ba0907185edd93c949ab20f34d21")
    p.add_argument("--no-headless", action="store_false")
    args = p.parse_args()
    
    # Load tasks from local directory
    tasks = load_local_tasks(args.filter)    
    print(f"Running {len(tasks)} tasks")
    
    # Run each task
    results = []
    for task in tasks:
        result = run_task(task, args.run_id, not args.no_headless)
        results.append(result)
    
    return results
    
    


if __name__ == "__main__":
    main()