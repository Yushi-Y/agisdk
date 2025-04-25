import dataclasses
from typing import Dict, Tuple, Optional

from agisdk import REAL


def run_builtin_agent():
    # Create a harness with the gpt-4o model
    harness = REAL.harness(
        model="gpt-4o-mini",
        task_name="webclones.omnizon-2",  # Specific task
        headless=False,                   # Show browser window
        max_steps=25,                     # Maximum steps per task
        use_screenshot=True,              # Include screenshots in observations
        use_axtree=True,                  # Include accessibility tree
    )
    
    # Run the task and get results
    results = harness.run()
    return results


if __name__ == "__main__":
    results = run_builtin_agent()  # run_leaderboard_submission()
    

# Example running multiple tasks with leaderboard submission
# def run_leaderboard_submission():
#     # Create a harness for leaderboard submission
#     harness = REAL.harness(
#         model="gpt-4o-mini",
#         leaderboard=True,
#         run_id="1e7a0bed-f6fa-483a-b304-1f4084187e7e",    # Your unique run ID for the leaderboard
#         # task_type="omnizon",       # Run all omnizon tasks
#         headless=True,             # Run headless for submissions
#         parallel=True,             # Run tasks in parallel
#         num_workers=20,             # Number of parallel workers
#     )
    
#     # Run tasks
#     results = harness.run()
#     return results


