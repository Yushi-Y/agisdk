# Tasks in AGI SDK

Tasks are the foundation of agent evaluation in the AGI SDK. They define browser-based challenges that AI agents must complete to demonstrate their capabilities.

## Task Structure

Tasks are defined in JSON format with the following structure:

```json
{
  "id": "unique-task-id",
  "goal": "Human-readable description of what the agent should accomplish",
  "website": {
    "id": "website-id",
    "name": "Website Name",
    "similarTo": "Real-world equivalent (e.g., LinkedIn, Amazon)",
    "previewImage": "/path/to/preview/image.jpg",
    "url": "https://website-url.example.com/"
  },
  "difficulty": "easy|medium|hard",
  "challengeType": "retrieval|generation|navigation",
  "possible": true,
  "evals": [
    {
      "description": "Evaluation criteria description",
      "type": "jmespath|llm_boolean",
      "query": "jmespath.query.string",
      "expected_value": "expected result"
    }
  ],
  "points": 1,
  "config": {}
}
```

### Key Fields

- **id**: Unique identifier for the task (e.g., "staynb-1", "omnizon-3")
- **goal**: Human-readable instructions describing what the agent should accomplish
- **website**: Information about the website the agent will interact with
  - **id**: Website identifier (e.g., "networkin", "omnizon")
  - **name**: Display name for the website
  - **similarTo**: Real-world equivalent (e.g., "LinkedIn", "Amazon")
  - **url**: URL where the website is hosted
- **difficulty**: Relative difficulty rating ("easy", "medium", "hard")
- **challengeType**: Category of challenge ("retrieval", "generation", "navigation")
- **evals**: Array of evaluation criteria that determine if the task was completed successfully
- **points**: Points awarded for completing the task

## Evaluation Mechanisms

Tasks are evaluated through two primary mechanisms:

### JMESPath Evaluations

JMESPath queries check specific values in the environment state:

```json
{
  "description": "Exactly one post was modified in the feed",
  "type": "jmespath",
  "query": "length(feedPostsDiff.modified)",
  "expected_value": 1
}
```

### LLM Boolean Evaluations

LLM evaluations use an AI model to determine if the agent's response meets criteria:

```json
{
  "description": "Email contains all required information",
  "type": "llm_boolean",
  "query": "Did the email contain all the requested information? The email should include...",
  "context_key": "emailContent"
}
```

## Task Examples

The repository contains numerous example tasks categorized by website type:

- E-commerce tasks (Omnizon): Finding products, adding to cart, checkout
- Travel booking (Staynb): Finding accommodations, making reservations
- Professional networking (Networkin): Posting updates, connecting with users
- Calendar management (GoCalendar): Scheduling appointments
- Email tasks (GoMail): Composing and sending emails

Example task file: [/example/tasks/test.json](/Users/pran-ker/Developer/agisdk/example/tasks/test.json)

More complex tasks can be found in: [/src/agisdk/REAL/browsergym/webclones/tasks/](/Users/pran-ker/Developer/agisdk/src/agisdk/REAL/browsergym/webclones/tasks/)

## Running Tasks

Tasks can be run using the harness module. Basic usage:

```python
from agisdk.REAL.harness import run_task

# Run a specific task with your agent
results = run_task("omnizon-1", my_agent)
```

For custom agent implementation, refer to the examples in the `/example` directory.

## Leaderboard and Submission

After developing and testing your agent against the available tasks, you can submit your results to the REAL benchmark leaderboard. Visit [https://realworldbenchmark.org](https://realworldbenchmark.org) for more information on submission.