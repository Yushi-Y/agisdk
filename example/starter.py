from agisdk import REAL


def run_builtin_agent(api_key=None, run_name=None, model_id_name=None):
    # Create a harness with the specified model
    # The harness will automatically get a run ID from the API if api_key and run_name are provided
    harness = REAL.harness(
        model="gpt-4o-mini",              # Model to use for the agent
        task_name="webclones.omnizon-2",  # Specific task
        headless=False,                   # Show browser window
        max_steps=25,                     # Maximum steps per task
        use_screenshot=True,              # Include screenshots in observations
        use_axtree=True,                  # Include accessibility tree
        api_key=api_key,                  # REAL API key for getting a run ID
        run_name=run_name,                # Human-readable name for this run
        model_id_name=model_id_name       # Model ID name for the API (can be different from agent model)
    )
    
    # Run the task and get results
    results = harness.run()
    return results


if __name__ == "__main__":
    # Example usage with API key and run name
    # Replace these with your actual values
    # api_key = ""  # REAL API key for getting a run ID
    # run_name = ""  # Human-readable name for this run
    # model_id_name = ""  # The model ID you're using
    
    # Run with or without API integration
    # results = run_builtin_agent()  # Simple run without API integration
    results = run_builtin_agent()  # Run with API integration
    
    # results = run_leaderboard_submission(api_key, run_name, model_id_name)

# Example running multiple tasks with leaderboard submission
# def run_leaderboard_submission(api_key=None, run_name=None, model_id_name="gpt-4o-mini"):
#     # Create a harness for leaderboard submission
#     # The harness will automatically get a run ID from the API if api_key and run_name are provided
#     harness = REAL.harness(
#         model=model_id_name,        # Model to use for the agent
#         leaderboard=True,           # Enable leaderboard submission
#         api_key=api_key,            # REAL API key for getting a run ID
#         run_name=run_name,          # Human-readable name for this run
#         model_id_name=model_id_name, # Model ID name for the API (can be different from agent model)
#         # If api_key and run_name are not provided, you can specify a run_id directly:
#         # run_id="1e7a0bed-f6fa-483a-b304-1f4084187e7e",
#         task_type="omnizon",       # Run all omnizon tasks
#         headless=True,             # Run headless for submissions
#         parallel=True,             # Run tasks in parallel
#         num_workers=20,            # Number of parallel workers
#     )
    
#     # Run tasks
#     results = harness.run()
#     return results

