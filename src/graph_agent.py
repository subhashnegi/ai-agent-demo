import os
import operator
from typing import TypedDict
from typing_extensions import Annotated
from langchain_anthropic import ChatAnthropic
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv


load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage],operator.add]


search_tool = TavilySearch(
    max_results=3,
    tavily_api_key=os.getenv("TAVILY_API_KEY")
)

tools=[search_tool]

llm = ChatAnthropic(
    model="claude-sonnet-4-5",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=1024
)

# binding tools tells Claude what tools exist and how to call them
# Claude can now say "I want to call search_tool with query X"
llm_with_tool = llm.bind_tools(tools)


#Nodes
def agent_node(state:AgentState) -> dict:
    """
    The brain of the agent.
    Receives current state, decides what to do next.
    Either calls a tool or gives final answer.
    """
    messages= state["messages"]
    response = llm_with_tool.invoke(messages)

    return {"messages": [response]}

def should_continue(state:AgentState) -> str:
    """
    Edge function — traffic controller of the graph.
    Decides which node to go to next.
    """
    last_message = state["messages"][-1]

    if last_message.tool_calls:
        # Claude requested a tool → go to tools node
        return "use_tools"

    # Claude has the final answer
    return END

# ToolNode is prebuilt — it automatically:
# 1. reads which tool Claude requested
# 2. runs that tool with the given arguments
# 3. returns the result as a ToolMessage
tool_node = ToolNode(tools)


#Build the graph______________

# Memory saves conversation state between turns
memory = MemorySaver()

# Create the graph with our state definition
graph = StateGraph(AgentState)

#Add Node

graph.add_node("agent",agent_node)
graph.add_node("tools",tool_node)

# Set where the graph starts
graph.set_entry_point("agent")

graph.add_conditional_edges(

    "agent",
    should_continue,
    {
        "use_tools": "tools",# if "use_tools" → tools node
        END: END
    }
)

# After tools always go back to agent
# This creates the ReAct loop
graph.add_edge("tools","agent")

# Compile with memory
agent = graph.compile(checkpointer=memory)

#Run the agent

def run_agent(question:str ,thread_id: str ="default") -> str:
    """
    Run the agent on a question.
    thread_id keeps conversations separate — like a session ID.
    """

    config = {"configurable": {"thread_id": thread_id}}

    result = agent.invoke(
        {"messages":[HumanMessage(content=question)]},
        config=config
    )
    return result["messages"][-1].content

def run_agent_with_history(thread_id:str = "default") -> None:
    """
    Interactive multi-turn conversation with memory.
    The agent remembers everything said in this thread.
    """
    print(f"Agent ready. Thread: {thread_id}")
    print("Type 'quit' to exit\n")

    while True:
        question = input("You: ").strip()

        if question.lower() == "quit":
            break

        if not question:
            continue

        print("Agent: thinking...", end="\r")
        response = run_agent(question, thread_id)
        print(f"Agent: {response}\n")


if __name__ == "__main__":

    print("=" * 50)
    print("Test 1: Question needing web search")
    print("=" * 50)

    response = run_agent(
        "Who won the Canada GradPrix 2026 and what date was the race?",
        thread_id="test-1"
    )
    print(f"Answer: {response}\n")

    # Test 2 — multi turn memory test
    print("=" * 50)
    print("Test 2: Memory test")
    print("=" * 50)

    run_agent("My name is Subhash and I am building an AI agent.", "memory-test")
    answer = run_agent("What is my Name and what am i building?", "memory-test")

    print(f"Memory test answer: {answer}\n")

    # Test 3 — interactive conversation
    print("=" * 50)
    print("Test 3: Interactive conversation")
    print("=" * 50)
    run_agent_with_history("interactive-session")

