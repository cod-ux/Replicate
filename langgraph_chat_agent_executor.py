import toml
import os
import phoenix as px
from phoenix.trace.langchain import LangChainInstrumentor


from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolExecutor
from langchain_openai import ChatOpenAI
from langchain.tools.render import format_tool_to_openai_function

from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage

from langgraph.prebuilt import ToolInvocation
from langchain_core.messages import FunctionMessage
import json
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage

secrets = "/Users/suryaganesan/Documents/GitHub/Replicate/secrets.toml"
github_secrets = "secrets.toml"

os.environ["OPENAI_API_KEY"] = toml.load(github_secrets)["OPENAI_API_KEY"]
os.environ["TAVILY_API_KEY"] = toml.load(github_secrets)["TAVILY_API_KEY"]



session = px.launch_app()
LangChainInstrumentor().instrument()

# Setup tools

tools = [TavilySearchResults(max_results=1)]
tool_exec = ToolExecutor(tools)

#Setup model

model = ChatOpenAI(model="gpt-4", temperature=0.7, streaming=True)

functions = [format_tool_to_openai_function(t) for t in tools]
model = model.bind_functions(functions)

# Define agent state

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

# Define nodes

def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]

    if "function_call" not in last_message.additional_kwargs:
        return "end"

    else:
        return "continue"

def call_model(state):
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}


def call_tool(state):
    messages = state["messages"]
    last_message = messages[-1]
    action = ToolInvocation(
        tool = last_message.additional_kwargs["function_call"]["name"],
        tool_input = json.loads(
            last_message.additional_kwargs["function_call"]["arguments"]
            ),
    )

    response = tool_exec.invoke(action)
    function_message = FunctionMessage(content=str(response), name=action.tool)
    return {"messages": [function_message]}

    
# Define Graph

workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("action", call_tool)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "action",
        "end": END
    },
)

workflow.add_edge("action", END)

app = workflow.compile()

# Using it!

print("no error till here ------")


msg = {"messages": [HumanMessage(content="Search oto find out what was the last IPL match that happened in Indian cricket")]}
response = app.invoke(msg)

print(response)