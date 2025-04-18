from agisdk import real
harness = REAL.harness(model="gpt-4o", leaderboard=True, run_id="1233_1233_4313_1342")
results = harness.run()

from typing import Union, Tuple, Dict


#######
from agisdk import REAL

class YourAgent(REAL.Agent):

    def __init__(self) -> None:
        super().__init__()
        self.answer_provided = False
        self.user_answer = None

        ### Implement your agent action here ###
    def get_agent_action(self, obs) -> Tuple[Union[str, None], Union[str, None]]:
    
        # To return an action without ending the episode
        return "```page.goto('https://www.google.com')```", None # Example action
    
        # To end the episode
        return None, "Task completed Successfully!" # Example ending 

    def get_action(self, obs: dict) -> Tuple[str, Dict]:
        
        # observation keys: dict_keys(
            # ['chat_messages', 'goal', 'goal_object',
            # 'open_pages_urls', 'active_page_index', 'url',
            # 'screenshot', 'dom_object', 'axtree_object',
            # 'extra_element_properties', 'focused_element_bid',
            # 'last_action', 'last_action_error', 'elapsed_time', 'browser'])

        agent_action, final_message = self.get_agent_action(obs) 

        if final_message:
            return f"send_msg_to_user({final_message})", {}
        else:
            return agent_action, {}


@dataclasses.dataclass
class YourAgentArgs(REAL.AbstractAgentArgs):
    agent_name: str = "AGI"
    
    def make_agent(self):
        return YourAgent()

harness = REAL.harness(
    agentargs=YourAgentArgs(agent_name="AGI"),
    leaderboard=True,
    run_id="1233_1233_4313_1342",
    )