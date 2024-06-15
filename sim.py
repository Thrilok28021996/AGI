from langchain.tools import DuckDuckGoSearchRun
from langchain.agents import (
    Tool,
    LLMSingleActionAgent,
    AgentExecutor,
)
from langchain import OpenAI, LLMChain
from langchain.memory import ConversationBufferWindowMemory
from utils import CustomPromptTemplate, CustomOutputParser
from Agent1_prompt import template
import os

os.environ["OPENAI_API_KEY"] = "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"
llm = OpenAI(temperature=0)


search = DuckDuckGoSearchRun()


# Next, let's load some tools to use. Note that the `llm-math` tool uses an LLM, so we need to pass that in.
tools = [
    Tool(
        name="search",
        func=search.run,
        description="useful for when you need to answer questions about current events.",
    )
]


prompt = CustomPromptTemplate(
    template=template,
    tools=tools,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["input", "intermediate_steps", "history"],
)


output_parser = CustomOutputParser()


## setup LLM
llm = OpenAI(temperature=0)


## setup the Agent
# LLM chain consisting of the LLM and a prompt
llm_chain = LLMChain(llm=llm, prompt=prompt)

tool_names = [tool.name for tool in tools]
agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=["\nObservation:"],
    allowed_tools=tool_names,
)


## Add memory
memory = ConversationBufferWindowMemory(k=2)

## use the Agent

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True, memory=memory
)

agent_executor.run(
    "How to use langchain and build agents such that the agents work together to complete the task?"
)
