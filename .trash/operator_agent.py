# Adapted from https://github.com/openai/openai-cua-sample-app/

import base64
import dataclasses
import os
import requests
import numpy as np
import io
import logging
from PIL import Image

from agisdk.REAL.browsergym.experiments import Agent, AbstractAgentArgs

logger = logging.getLogger(__name__)


def image_to_jpg_base64_url(image: np.ndarray | Image.Image):
    """Convert a numpy array to a base64 encoded image url."""

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if image.mode in ("RGBA", "LA"):
        image = image.convert("RGB")

    with io.BytesIO() as buffer:
        image.save(buffer, format="JPEG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode()

    return f"data:image/jpeg;base64,{image_base64}"

class OperatorAgent(Agent):
    """A basic agent using OpenAI's Operator agent API (computer-use-preview), to demonstrate BrowserGym's functionalities."""

    @staticmethod
    def create_response(**kwargs):
        url = "https://api.openai.com/v1/responses"
        headers = {
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
            "Content-Type": "application/json"
        }

        openai_org = os.getenv("OPENAI_ORG")
        if openai_org:
            headers["Openai-Organization"] = openai_org

        response = requests.post(url, headers=headers, json=kwargs)

        if response.status_code != 200:
            # Log error instead of just printing
            logger.error(f"API Error: {response.status_code} {response.text}")
            # Consider returning an error indicator or raising exception
            return {"error": response.text, "status_code": response.status_code}

        try:
            return response.json()
        except requests.exceptions.JSONDecodeError:
            logger.error(f"API Error: Failed to decode JSON response: {response.text}")
            return {"error": "Failed to decode JSON response", "status_code": response.status_code}

    def __init__(
        self,
        chat_mode: bool, # Note: chat_mode is not explicitly used in the CUA logic currently
        demo_mode: str, # Note: demo_mode is not explicitly used in the CUA logic currently
        use_html: bool, # Note: use_html is not explicitly used in the CUA logic currently
        use_axtree: bool, # Note: use_axtree is not explicitly used in the CUA logic currently
        use_screenshot: bool, # Note: use_screenshot is ignored, screenshot is always sent
        browser_dimensions: tuple[int, int], # (width, height)
        model_name: str = "computer-use-preview",
        backend: str = "openai", # Note: backend param not used, hardcoded openai url
    ) -> None:
        super().__init__()
        self.model_name = model_name
        # Store browser dimensions for the tool definition
        self.browser_dimensions = {
            "viewport_width": browser_dimensions[0],
            "viewport_height": browser_dimensions[1],
        }
        # The CUA agent doesn't use the BrowserGym action_set concept directly
        self.action_set = None

        # State for multi-turn interaction
        self.last_response_id = None
        self.last_call_id = None
        # Store pending safety checks
        self.pending_safety_checks = []


    def get_action(self, obs: dict) -> tuple[dict, dict]: # Return action dict and info dict
        """Gets the next action from the OpenAI CUA model."""

        # ---- Define the CUA tool ----
        tools = [
            {
                "type": "computer_use_preview", # Note: API docs use computer_use_preview now
                "display_width": self.browser_dimensions['viewport_width'],
                "display_height": self.browser_dimensions['viewport_height'],
                "environment": "browser",
            }
        ]

        # ---- Get current screenshot ----
        screenshot_url = None
        if "screenshot" in obs and obs["screenshot"] is not None:
            screenshot_url = image_to_jpg_base64_url(obs["screenshot"])
        else:
            logger.error("Screenshot missing in observation!")
            return {"type": "error", "message": "Missing screenshot in observation"}, {}


        # ---- Prepare API call based on turn ----
        request_body = {"model": self.model_name, "tools": tools, "truncation": "auto"}

        if self.last_response_id is None:
            # ---- FIRST TURN ----
            input_items = []

            # Extract goal
            if "goal_object" in obs and isinstance(obs["goal_object"], list) and len(obs["goal_object"]) > 0:
                # Assuming the first text message is the primary goal
                goal_statement = obs["goal_object"][0].get("text", "No goal text found.")
                input_items.append({"role": "system", "content": """
You are a helpful assistant that can help with tasks in a web browser.
The task provided needs to be completed without any human intervention. Don't ever ask the user for any input! Don't ask the user for verification, just do it!
You can execute only one action at a time. This is a test environment, so don't worry about the cost of the actions.
Once you have completed the task, you can send a message to the user saying "Done" or returning any information that is requested in the following task.
                                    """})
                input_items.append({"role": "user", "content": goal_statement})
            else:
                logger.warning(f"Goal missing or invalid format in obs: {obs.get('goal_object')}")
                raise ValueError("Goal missing or invalid format in obs")

            # Add initial screenshot
            input_items.append({
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": screenshot_url,
                    },
                ],
            })

            request_body["input"] = input_items
            request_body["reasoning"] = {"generate_summary": "concise"} # Add reasoning for first turn

        else:
            # ---- SUBSEQUENT TURNS ----
            if self.last_call_id is None:
                logger.error("Cannot make subsequent call: last_call_id is missing.")
                return {"type": "error", "message": "Missing last_call_id for subsequent turn"}, {}

            request_body["previous_response_id"] = self.last_response_id
            
            # Prepare input with acknowledged safety checks if any
            input_data = {
                "call_id": self.last_call_id,
                "type": "computer_call_output",
                "output": {
                    "type": "input_image",
                    "image_url": screenshot_url
                }
            }
            system_prompt = {"role": "system", "content": """
You are a helpful assistant that can help with tasks in a web browser.
The task provided needs to be completed without any human intervention. Don't ever ask the user for any input! Don't ask the user for verification, just do it!
You can execute only one action at a time. This is a test environment, so don't worry about the cost of the actions.
Once you have completed the task, you can send a message to the user saying "Done" or returning any information that is requested in the task.
                                    """}
            # Automatically acknowledge any pending safety checks
            if self.pending_safety_checks:
                print(f"Acknowledging {len(self.pending_safety_checks)} safety checks")
                input_data["acknowledged_safety_checks"] = self.pending_safety_checks
                self.pending_safety_checks = []  # Clear after acknowledging
                
            request_body["input"] = [input_data]
            request_body["input"].append(system_prompt)

        # ---- Call the API ----
        # Create a copy of request_body to avoid modifying the original
        import copy
        log_body = copy.deepcopy(request_body)
        
        # Remove image data from logging if present
        def remove_image_urls(obj):
            if isinstance(obj, dict):
                for key, value in list(obj.items()):
                    if key == "image_url":
                        obj[key] = "[IMAGE_DATA_REMOVED]"
                    else:
                        remove_image_urls(value)
            elif isinstance(obj, list):
                for item in obj:
                    remove_image_urls(item)
        
        remove_image_urls(log_body)
        
        # print(f"Request Body: {json.dumps(log_body, indent=2)}") # Verbose logging without image data
        response = self.create_response(**request_body)
        # print(f"Response: {json.dumps(response, indent=2)}") # Verbose logging

        # ---- Process response ----
        # Check for API errors reported by create_response
        if "error" in response and response["error"] is not None:
            logger.error(f"API call failed: {response['error']}")
            return {"type": "error", "message": f"API Error: {response.get('error', 'Unknown')}"}, {}

        # Store response ID for the next turn
        self.last_response_id = response["id"]

        # Find the computer action and call_id
        action_to_execute = None
        new_call_id = None
        for item in response["output"]: # Only use the last item in the output list
            if item.get("type") == "computer_call":
                new_call_id = item.get("call_id")
                action_to_execute = item.get("action")
                
                # Store any pending safety checks for automatic acknowledgment in next turn
                if "pending_safety_checks" in item and item["pending_safety_checks"]:
                    self.pending_safety_checks = item["pending_safety_checks"]
                break

        # Store the call_id for the next turn
        self.last_call_id = new_call_id
        if action_to_execute:
            return action_to_execute, {}
        else:
            # Try finding completed task message or raise error
            for item in response["output"]:
                if item.get("type") == "message":
                    content = item.get("content", [])
                    if content and isinstance(content, list) and len(content) > 0:
                        message = content[0].get("text", "Task completed with no message")
                        return {"type": "message", "content": message}, {}
            
            return {"type": "error", "message": "No action or message found in response"}, {}


@dataclasses.dataclass
class OperatorAgentArgs(AbstractAgentArgs):
    """
    Arguments for OperatorAgent. Note that some parameters inherited from
    AbstractAgentArgs might not be directly used by the CUA implementation.
    """
    model_name: str = "computer-use-preview" # Default to the correct model
    # chat_mode, demo_mode, use_html, use_axtree might be less relevant for CUA
    chat_mode: bool = False
    demo_mode: str = "off"
    use_html: bool = False
    use_axtree: bool = False # CUA likely relies on screenshot primarily
    use_screenshot: bool = True # Keep True, logic forces it anyway
    backend: str = "openai" # Currently hardcoded in create_response

    # Need to accept browser_dimensions in make_agent
    def make_agent(self, browser_dimensions: tuple[int, int]):
        """Creates an instance of the OperatorAgent."""
        return OperatorAgent(
            # Pass all args, even if some are noted as potentially unused by current logic
            model_name=self.model_name,
            chat_mode=self.chat_mode,
            demo_mode=self.demo_mode,
            use_html=self.use_html,
            use_axtree=self.use_axtree,
            use_screenshot=self.use_screenshot,
            browser_dimensions=browser_dimensions, # Pass dimensions
            backend=self.backend,
        )
