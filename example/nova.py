# Example of using Amazon NovaAct to perform a task on a website
from nova_act import NovaAct
import urllib.parse



import os
import json

task_dir = "../src/agisdk/REAL/browsergym/webclones/tasks"
tasks = []

for filename in os.listdir(task_dir):
    if filename.endswith(".json"):
        filepath = os.path.join(task_dir, filename)
        with open(filepath, "r") as f:
            tasks.append(json.load(f))

run_id = "9445b200-1656-4899-bdf4-217950afaa82"

for task in tasks:
    if not task["id"] == "dashdish-1":
        continue
    goal = task["goal"]
    url = task["website"]["url"]
    task_id = task["id"]
    config_url = url + f"/config?run_id={run_id}&task_id={task_id}"
    print(f"Goal: {goal}")
    with NovaAct(starting_page=config_url) as nova:
        nova.go_to_url(url)
        result = nova.act(goal)
        response = result.response if result.response is not None else "No response"
        print(f"Response: {response}")
        encoded_response = urllib.parse.quote(response)
        nova.go_to_url(url + "/submit?retrieved_answer=" + encoded_response)
        print(f"Submitted response to {url + '/submit?retrieved_answer=' + encoded_response}")
