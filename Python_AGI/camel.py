## CAMEL Role-Playing Autonomous Cooperative Agents
## CAMEL: Communicative Agents for “Mind” Exploration of Large Scale Language Model Society

## Import LangChain related modules

from langchain.schema import (
    SystemMessage,
    HumanMessage,
)

from langchain.prompts.chat import (
    HumanMessagePromptTemplate,
)
from langchain.chat_models import ChatOpenAI
from camelagent import CAMELAgent

from dotenv import load_dotenv

load_dotenv()

from utils import get_sys_msgs
import colorama

###  Setup OpenAI API key and roles and task for role-playing
import os


assistant_role_name = "Agent007"
user_role_name = "Agent369"
task = """ write the code for fibonacci series."""


word_limit = 50  # word limit for task brainstorming


### Create a task specify agent for brainstorming and get the specified task
task_specifier_sys_msg = SystemMessage(content="You can make a task more specific.")
task_specifier_prompt = """Here is a task that {assistant_role_name} will help {user_role_name} to complete: {task}.
Please make it more specific. Be creative and imaginative.
Please reply with the specified task in {word_limit} words or less. Do not add anything else."""
task_specifier_template = HumanMessagePromptTemplate.from_template(
    template=task_specifier_prompt
)
task_specify_agent = CAMELAgent(task_specifier_sys_msg, ChatOpenAI(temperature=1.0))
task_specifier_msg = task_specifier_template.format_messages(
    assistant_role_name=assistant_role_name,
    user_role_name=user_role_name,
    task=task,
    word_limit=word_limit,
)[0]
specified_task_msg = task_specify_agent.step(task_specifier_msg)
print(
    colorama.Fore.MAGENTA
    + f"Specified task: {specified_task_msg.content}"
    + colorama.Style.RESET_ALL
)
specified_task = specified_task_msg.content


### Create AI assistant agent and AI user agent from obtained system messages
assistant_sys_msg, user_sys_msg = get_sys_msgs(
    assistant_role_name, user_role_name, specified_task
)
assistant_agent = CAMELAgent(assistant_sys_msg, ChatOpenAI(temperature=0.2))
user_agent = CAMELAgent(user_sys_msg, ChatOpenAI(temperature=0.2))

# Reset agents
assistant_agent.reset()
user_agent.reset()

# Initialize chats
assistant_msg = HumanMessage(
    content=(
        f"{user_sys_msg.content}. "
        "Now start to give me introductions one by one. "
        "Only reply with Response,Reflections, code and Critique"
    )
)

user_msg = HumanMessage(content=f"{assistant_sys_msg.content}")
user_msg = assistant_agent.step(user_msg)


### Start role-playing session to solve the task!
print(colorama.Fore.RED + f"Original task prompt:\n{task}\n" + colorama.Style.RESET_ALL)
print(
    colorama.Fore.CYAN
    + f"Specified task prompt:\n{specified_task}\n"
    + colorama.Style.RESET_ALL
)
chat_turn_limit, n = 30, 0
while n < chat_turn_limit:
    n += 1
    user_ai_msg = user_agent.step(assistant_msg)
    user_msg = HumanMessage(content=user_ai_msg.content)
    print(
        colorama.Fore.BLUE
        + f"AI User ({user_role_name}):\n\n{user_msg.content}\n\n"
        + colorama.Style.RESET_ALL
    )

    assistant_ai_msg = assistant_agent.step(user_msg)
    assistant_msg = HumanMessage(content=assistant_ai_msg.content)
    print(
        colorama.Fore.GREEN
        + f"AI Assistant ({assistant_role_name}):\n\n{assistant_msg.content}\n\n"
        + colorama.Style.RESET_ALL
    )
    if "<TASK_DONE>" in user_msg.content:
        break
