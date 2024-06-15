from langchain.tools import DuckDuckGoSearchRun
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI
from langchain.agents import Tool


import os

os.environ["OPENAI_API_KEY"] = "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"


search = DuckDuckGoSearchRun()

llm = OpenAI(temperature=0)


# Next, let's load some tools to use. Note that the `llm-math` tool uses an LLM, so we need to pass that in.
tools = [
    Tool(
        name="search",
        func=search.run,
        description="useful for when you need to answer questions about current events. You should ask targeted questions",
    )
]


# Finally, let's initialize an agent with the tools, the language model, and the type of agent we want to use.
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

# Now let's test it out!
print(agent.run("Who is Olivia Wilde's boyfriend? What is his current age?"))
