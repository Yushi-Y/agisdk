from . import browsergym
from .harness import harness, Agent, AbstractAgentArgs

def hello(name="World"):
    """A real greeting function for the real submodule."""
    message = f"Hello {name}, from the real world!"
    print(message)
    return message