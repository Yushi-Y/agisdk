import json
import os
from pathlib import Path

# Load all task JSON files
tasks_dir = Path(__file__).parent / "browsergym" / "webclones" / "tasks"

# Create a flat list of all tasks
all_tasks = []

for file_path in tasks_dir.glob("*.json"):
    # Load JSON and add to list
    with open(file_path, "r") as f:
        task = json.load(f)
        # Add filename to task for reference
        task["_filename"] = file_path.name
        all_tasks.append(task)