import os
import toml
import asyncio
import phoenix as px
from phoenix.trace.langchain import LangChainInstrumentor

from langchain_community.tools.tavily_search import TavilySearchResults

from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_agent_executor

from langchain_core.pydantic_v1 import BaseModel, Field
from typing import TypedDict, List, Tuple, Annotated
import operator

from langchain.chains.openai_functions import create_structured_output_runnable, create_openai_fn_runnable
from langchain_core.prompts import ChatPromptTemplate

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage

# setup

secrets = "/Users/suryaganesan/Documents/GitHub/Replicate/secrets.toml"
github_secrets = "secrets.toml"

os.environ["OPENAI_API_KEY"] = toml.load(github_secrets)["OPENAI_API_KEY"]
os.environ["TAVILY_API_KEY"] = toml.load(github_secrets)["TAVILY_API_KEY"]

session = px.launch_app()
LangChainInstrumentor().instrument()


# Define tools

tools = [TavilySearchResults(max_results=3)]

print("....Finished setting up")

# Define execution agent

prompt = hub.pull("hwchase17/openai-functions-agent")
llm = ChatOpenAI(model="gpt-4-turbo-preview", streaming=True)
agent_runnable = create_openai_functions_agent(llm, tools, prompt)
agent_executor = create_agent_executor(agent_runnable, tools)

print("Defining State, Plan....")

# Define state

class PlanExec_State(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str

# Define Planner

class Plan(BaseModel):
    steps: List[str] = Field(
        description="different steps to follow should be in sorted order"
        )

planner_prompt = ChatPromptTemplate.from_template(
    """For the given objective, come up with a simple step by step plan. \
This plan should involve individual takss, that if executed should yeild to the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

{objective}"""
)

planner = create_structured_output_runnable(Plan, ChatOpenAI(model="gpt-4-turbo-preview", temperature=0, streaming=True), planner_prompt)

class Response(BaseModel):
    response: str

replanner_prompt = ChatPromptTemplate.from_template(
    """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

Your objective was this:
{input}

Your original plan was this:
{plan}

You have currently done the following steps:
{past_steps}

Update your plan accordingly. If there are no more steps needed and you can retrun to the user, then respond with that. Otherwise fill out the plan. Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan."""
)

replanner = create_openai_fn_runnable(
    [Plan, Response], 
    ChatOpenAI(model="gpt-4-turbo-preview", temperature=0, streaming=True),
    replanner_prompt,
    )

# Create Graph

print("Defining Graph functions....")

async def execute_step(state: PlanExec_State):
    print("\nExecution agent begins...\n")
    task = state["plan"][0]
    print(f"Task retrieved: {task}")
    agent_response = await agent_executor.ainvoke({"input": task, "chat_history": []})
    print(f"Agent response recieved: {agent_response}")

    return {
        "past_steps": (task, agent_response["agent_outcome"].return_values["output"])
    }

async def plan_step(state: PlanExec_State):
    print("\nPlanning step begins...\n")
    plan = await planner.ainvoke({"objective": state["input"]})
    print("Planning step: ")
    return {"plan": plan.steps}

async def replan_step(state: PlanExec_State):
    print("\nReplanning step begins...\n")
    output = await replanner.ainvoke(state)
    if isinstance(output, Response):
        return {"response": output.response}
    else:
        return {"plan": output.steps}

def should_end(state: PlanExec_State):
    print("\nChecking if program should end...\n")
    if "response" in state and state["response"]:
        return "True"
    else:
        return "False"

# The Graph

print("Creating the Graph....")

workflow = StateGraph(PlanExec_State)

workflow.add_node("Executor", execute_step)
workflow.add_node("Planner", plan_step)
workflow.add_node("Replanner", replan_step)

workflow.set_entry_point("Planner")

workflow.add_edge("Planner", "Executor")
workflow.add_edge("Executor", "Replanner")

workflow.add_conditional_edges(
    "Replanner",
    should_end,
    {
        "True": END,
        "False": "Executor"
    }
)

print("Compiling the Graph....")

app = workflow.compile()

# Use it!

print("Executing request....\n\n")

config = {"recursion_limit": 25}
inputs = {"input": "How old was the queen Elizabeth of England when she died and How long was she queen?"}

async def main():
  async for event in app.astream(inputs, config=config):
        for key, value in event.items():
            if key != "__end__":
                print(value)

asyncio.run(main())
