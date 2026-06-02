import os
from typing import TypedDict, Annotated
from langchain_anthropic import ChatAnthropic
from langchain_tavily import TavilySearch
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    BaseMessage,
    SystemMessage
)
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
import operator

from rag_pipeline import build_rag_pipeline

load_dotenv()

# ── Initialise RAG ────────────────────────────────────────────────
# force_rebuild=False → loads existing index from disk
# fast startup, no Voyage AI calls needed

print("Initialising RAG pipeline...")
vector_store, embeddings = build_rag_pipeline(force_rebuild=False)
print("RAG pipeline ready\n")


# ── Shared State ──────────────────────────────────────────────────
# This is the COMMUNICATION CHANNEL between agents
# Both planner and executor read and write to this
# Like a shared whiteboard in an office

class MultiAgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    # ↑ full conversation history
    # operator.add = append new messages, never replace
    # both agents add their messages here

    task_plan: str
    # ↑ planner writes the full step by step plan here
    # executor reads it to understand the big picture
    # plain string — just text

    current_step: str
    # ↑ planner writes the SPECIFIC next step here
    # executor reads this to know exactly what to do
    # only one step at a time — keeps executor focused

    completed_steps: Annotated[list[str], operator.add]
    # ↑ executor writes completed steps here
    # operator.add = append each completed step
    # planner reads this to know what's already done
    # prevents repeating steps

    next_agent: str
    # ↑ controls who goes next
    # planner writes "executor" when step is ready
    # planner writes "end" when all steps complete
    # routing functions read this to decide graph flow


# ── Tools ─────────────────────────────────────────────────────────
# Only the executor uses tools
# Planner only thinks — never touches tools directly

# Tool 1 — Web search
# Used when executor needs current public information
web_search_tool = TavilySearch(
    max_results=3,
    tavily_api_key=os.getenv("TAVILY_API_KEY")
    # max_results=3 → return top 3 results
    # more results = more context but more tokens
    # 3 is good balance for our use case
)

# Tool 2 — Company document search (RAG)
# Used when executor needs internal TechCorp information
@tool
def search_company_docs(query: str) -> str:
    """
    Search TechCorp internal company documents.
    Use for any questions about TechCorp policies,
    pricing, support, data privacy, or API limits.
    Do NOT use for general knowledge or current events.
    """
    results = vector_store.similarity_search(query, k=3)

    if not results:
        return "No relevant information found in company documents."

    return "\n\n".join([
        f"Chunk {i+1}:\n{doc.page_content}"
        for i, doc in enumerate(results)
    ])

# executor_tools = tools available to executor agent only
# planner has NO tools — only reasoning
executor_tools = [web_search_tool, search_company_docs]

# ── Two separate LLM instances ────────────────────────────────────
# We create TWO Claude instances with different configurations
# This is what makes them behave as different agents

llm = ChatAnthropic(
    model="claude-sonnet-4-5",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=1024
)

# Planner LLM — no tools bound
# Planner only thinks and plans
# giving it tools would confuse it — it might try to execute
# instead of just planning
planner_llm = llm


# Executor LLM — tools bound
# Executor can call web search and RAG
# bind_tools tells Claude what tools are available
# Claude can now choose to call them
executor_llm = llm.bind_tools(executor_tools)


# ── System Prompts ────────────────────────────────────────────────
# System prompts define each agent's PERSONA and RULES
# This is what makes them behave differently
# even though they're the same base model

PLANNER_PROMPT = """You are a Planner Agent — a strategic manager.

YOUR ONLY JOB:
1. Analyse the user's question
2. Break it into clear numbered steps
3. Identify which step to execute NEXT
4. Review completed steps and decide what's still needed
5. When all steps complete — write the final answer

RULES:
- Never execute steps yourself — only plan
- Keep each step specific and actionable
- One step at a time — don't skip ahead
- Always check completed_steps before planning next step
- When writing final answer start with FINAL ANSWER:

FORMAT your response EXACTLY like this:
PLAN:
1. [step 1]
2. [step 2]
3. [step 3]

NEXT STEP: [exact step to execute now]
STATUS: [PLANNING / IN_PROGRESS / COMPLETE]"""


EXECUTOR_PROMPT = """You are an Executor Agent — a specialist worker.

YOUR ONLY JOB:
1. Read the specific step given to you
2. Use the available tools to complete it
3. Report back with clear factual results

RULES:
- Execute ONLY the specific step given — nothing else
- Always use tools — never answer from memory
- Be precise and factual in your results
- Do not plan or decide next steps
- Do not write final answers — just report results"""


# ── Planner Node ──────────────────────────────────────────────────
# The manager — thinks, plans, reviews, decides
# Never calls tools directly

def planner_node(state: MultiAgentState) -> dict:
    """
    Planner agent node.
    Reads current state and decides what step to do next.
    Writes plan and current_step to state.
    """
    # Get all messages so far
    messages = state["messages"]

    # Get what's already been completed
    # so planner doesn't repeat finished steps
    completed = state.get("completed_steps", [])

    # Build context about what's done
    # this goes into planner's prompt
    completed_context = ""
    if completed:
        completed_context = "\n\nCompleted steps:\n" + \
                            "\n".join(f"✓ {step}" for step in completed)

    # Invoke planner LLM with:
    # 1. system prompt (persona and rules)
    # 2. conversation history (what user asked)
    # 3. context about completed steps
    response = planner_llm.invoke([
        SystemMessage(content=PLANNER_PROMPT),
        *messages,
        # *messages = unpacks the list
        # like spreading all messages into the list
        # same as: messages[0], messages[1], messages[2]...
        HumanMessage(content=f"Plan the next step.{completed_context}")
    ])

    content = response.content

    # Parse planner's response to extract:
    # 1. next step to execute
    # 2. current status
    next_step = ""
    status = "IN_PROGRESS"

    for line in content.split("\n"):
        if line.startswith("NEXT STEP:"):
            next_step = line.replace("NEXT STEP:", "").strip()
        if line.startswith("STATUS:"):
            status = line.replace("STATUS:", "").strip()

    # Decide who goes next based on status
    if status == "COMPLETE" or not next_step:
        next_agent = "end"
    else:
        next_agent = "executor"

    # Write to shared state
    # executor will read current_step and next_agent
    return {
        "messages": [AIMessage(content=f"[PLANNER]: {content}")],
        "task_plan": content,
        "current_step": next_step,
        "next_agent": next_agent
    }

# ── Executor Node ─────────────────────────────────────────────────
# The employee — reads one step, uses tools, reports back

def executor_node(state: MultiAgentState) -> dict:
    """
    Executor agent node.
    Reads current_step from state.
    Uses tools to complete it.
    Writes result and completed step back to state.
    """
    # Read the specific step planner assigned
    current_step = state.get("current_step", "")
    messages = state["messages"]

    # Invoke executor LLM with:
    # 1. executor system prompt (worker persona)
    # 2. conversation history
    # 3. specific step to execute
    response = executor_llm.invoke([
        SystemMessage(content=EXECUTOR_PROMPT),
        *messages,
        HumanMessage(content=f"Execute this step: {current_step}")
    ])

    # If Claude wants to call a tool
    if response.tool_calls:
        # ToolNode runs the requested tool
        tool_node_instance = ToolNode(executor_tools)

        # Run the tool
        tool_result = tool_node_instance.invoke(
            {"messages": [*messages, response]}
        )
        tool_messages = tool_result["messages"]

        # Get final response from executor
        # after reading tool results
        final_response = executor_llm.invoke([
            SystemMessage(content=EXECUTOR_PROMPT),
            *messages,
            response,
            *tool_messages,
            HumanMessage(content="Summarise what you found.")
        ])

        # Write results back to state
        # completed_steps uses operator.add → appends
        return {
            "messages": [
                response,
                *tool_messages,
                AIMessage(content=f"[EXECUTOR]: {final_response.content}")
            ],
            "completed_steps": [current_step]
            # ↑ marks this step as done
            # planner reads this next time
        }

    # If Claude answered without tools
    return {
        "messages": [
            AIMessage(content=f"[EXECUTOR]: {response.content}")
        ],
        "completed_steps": [current_step]
    }

# ── Routing Functions ─────────────────────────────────────────────
# These are the traffic controllers
# they read state and decide where graph goes next

def route_after_planner(state: MultiAgentState) -> str:
    """
    Called after planner node runs.
    Reads next_agent from state.
    Returns which node to go to next.

    next_agent = "executor" → go to executor node
    next_agent = "end"      → stop the graph
    """
    next_agent = state.get("next_agent", "executor")

    if next_agent == "end":
        return END
    return "executor"


def route_after_executor(state: MultiAgentState) -> str:
    """
    Called after executor node runs.
    Always goes back to planner.
    Planner reviews results and decides next step.

    No conditional logic here — always back to planner.
    This creates the planner → executor → planner loop.
    """
    return "planner"


# ── Build Multi-Agent Graph ───────────────────────────────────────

memory = MemorySaver()
# MemorySaver saves MultiAgentState between invocations
# same concept as in rag_agent.py
# keyed by thread_id

graph = StateGraph(MultiAgentState)
# StateGraph takes our state definition
# knows what fields to track
# knows operator.add means append not replace

# Add both agent nodes
graph.add_node("planner", planner_node)
graph.add_node("executor", executor_node)
# "planner" and "executor" are just names
# they must match what we use in routing functions

# Always start with planner
# planner analyses the question first
# then decides what executor should do
graph.set_entry_point("planner")

# After planner — conditional routing
# route_after_planner decides: executor or END
graph.add_conditional_edges(
    "planner",              # from planner node
    route_after_planner,    # call this function
    {
        "executor": "executor",  # "executor" → go to executor node
        END: END                 # END → stop graph
    }
)

# After executor — always back to planner
# no conditions — planner always reviews
graph.add_conditional_edges(
    "executor",             # from executor node
    route_after_executor,   # call this function
    {
        "planner": "planner"     # always back to planner
    }
)

multi_agent = graph.compile(checkpointer=memory)
# compile = finalise the graph, make it runnable
# checkpointer=memory = save state between calls

# ── Run Function ──────────────────────────────────────────────────

def run_multi_agent(question: str, thread_id: str = "default") -> str:
    """
    Run the multi-agent system on a complex question.

    question  = what you want to research
    thread_id = conversation ID for memory isolation
    """
    config = {"configurable": {"thread_id": thread_id}}

    print("\n" + "="*60)
    print(f"Multi-Agent Task: {question}")
    print("="*60)

    # Invoke graph with initial state
    # ALL fields of MultiAgentState must be provided
    # empty defaults for fields that start empty
    result = multi_agent.invoke(
        {
            "messages": [HumanMessage(content=question)],
            "task_plan": "",          # planner fills this
            "current_step": "",       # planner fills this
            "completed_steps": [],    # executor fills this
            "next_agent": "executor"  # default — goes to executor first
        },
        config=config
    )

    # Show the full agent conversation
    # so we can see planner and executor talking
    print("\n--- Agent conversation ---")
    for msg in result["messages"]:
        if isinstance(msg, AIMessage) and msg.content:
            if "[PLANNER]" in msg.content:
                print(f"\n🧠 PLANNER:")
                print(msg.content.replace("[PLANNER]:", "").strip()[:300])
                print("...")
            elif "[EXECUTOR]" in msg.content:
                print(f"\n⚙️  EXECUTOR:")
                print(msg.content.replace("[EXECUTOR]:", "").strip()[:300])
                print("...")

    # Show completed steps
    print(f"\n--- Completed steps ---")
    for i, step in enumerate(result.get("completed_steps", [])):
        print(f"  {i+1}. {step}")

    # Return final answer — last meaningful AI message
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            if "[PLANNER]" in msg.content and "FINAL ANSWER" in msg.content:
                # Extract just the final answer part
                content = msg.content
                if "FINAL ANSWER:" in content:
                    return content.split("FINAL ANSWER:")[1].strip()
                return content.replace("[PLANNER]:", "").strip()

    return result["messages"][-1].content


if __name__ == "__main__":
    answer = run_multi_agent(
        "What is TechCorp's refund policy and who won the 2026 Spelling Bee?",
        thread_id="multi-agent-test-2"
    )