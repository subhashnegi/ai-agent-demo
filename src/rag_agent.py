import os
from typing import TypedDict, Annotated
from langchain_anthropic import ChatAnthropic
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
import operator

from rag_pipeline import build_rag_pipeline

# Add system prompt at the top of your file after imports
SYSTEM_PROMPT = """You are a helpful TechCorp customer support assistant.

You have access to two tools:
1. search_company_docs — for TechCorp internal policies and information
2. tavily_search — for current public information

IMPORTANT RULES:
- Always use search_company_docs for questions about TechCorp policies
- Always use tavily_search for current events and public information
- After getting tool results, ALWAYS write a clean summarised answer
- Never return raw tool results as your final answer
- Be concise and helpful in your responses
- Always synthesise tool results into a clean readable answer
- Never return raw JSON or tool output directly to the user
- If search returns results always extract the key facts
  and present them clearly"""

load_dotenv()

# ── Initialise RAG pipeline ───────────────────────────────────────
# This runs when the file is imported
# builds or loads the Qdrant vector store
# force_rebuild=False → loads from disk if exists (fast)

print("Initialising RAG pipeline...")
vector_store, embeddings = build_rag_pipeline(force_rebuild=False)
print("RAG pipeline ready\n")

# ── State ─────────────────────────────────────────────────────────
#  messages accumulate across all turns

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]

# ── Tool 1: Web search ────────────────────────────────────────────
# For public internet questions

web_search_tool = TavilySearch(
    max_results=3,
    tavily_api_key=os.getenv("TAVILY_API_KEY")
)

# ── Tool 2: Company document search (RAG) ─────────────────────────
# For internal TechCorp questions
# THIS IS NEW — converting RAG into a tool Claude can call

@tool
def search_company_docs(query: str) -> str:
    """
    Search TechCorp internal company documents.
    Use this tool for questions about:
    - refund policy
    - subscription plans and pricing
    - support hours and SLAs
    - data privacy policy
    - API usage limits
    Do NOT use for general knowledge or current events.
    Use web_search for those instead.
    """
    # Search Qdrant for relevant chunks
    results = vector_store.similarity_search(query, k=3)

    if not results:
        return "No relevant information found in company documents."

    # Combine top chunks into one string for Claude
    # Claude reads this and formulates a proper answer
    combined = "\n\n".join([
        f"Document chunk {i+1}:\n{doc.page_content}"
        for i, doc in enumerate(results)
    ])

    return combined

# ── Tools list ────────────────────────────────────────────────────
# Both tools available to Claude
# Claude decides which to call based on the question
tools =[web_search_tool,search_company_docs]

# ── LLM with tools bound ──────────────────────────────────────────

llm = ChatAnthropic(
    model="claude-sonnet-4-5",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=1024
)

llm_with_tools = llm.bind_tools(tools)
# ── Nodes ─────────────────────────────────────────────────────────


def agent_node(state: AgentState) -> dict:
    """
    The brain — Claude decides what to do.
    Reads all messages so far and either:
    1. Calls a tool (web search or company docs)
    2. Gives final answer (no more tools needed)
    """
    messages = state["messages"]
    # Prepend system prompt to every invocation
    all_messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def should_continue(state:AgentState) -> str:
    """
    Traffic controller — decides where to go next.
    Tool calls present → go to tools node
    No tool calls      → end, Claude has final answer
    """
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "use_tools"

    return END

tool_node = ToolNode(tools)

memory = MemorySaver()
graph = StateGraph(AgentState)

graph.add_node("agent",agent_node)
graph.add_node("tools",tool_node)

graph.set_entry_point("agent")

graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "use_tools":"tools",
        END:END

    }
)
graph.add_edge("tools","agent")
agent = graph.compile(checkpointer=memory)


def run_agent(question: str, thread_id: str = "default") -> str:
    """
    Run the RAG agent on a question.
    Shows which tool was used for transparency.
    """
    config = {"configurable": {"thread_id": thread_id}}

    print("\n" + "="*50)
    print(f"Question: {question}")
    print("="*50)

    result = agent.invoke(
        {"messages": [HumanMessage(content=question)]},
        config=config
    )
    # Show which tools were called — like a trace
    print("\n--- Tools used ---")
    tool_used = False
    for msg in result["messages"]:
        msg_type = type(msg).__name__

        if msg_type == "AIMessage" and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"  Tool called: {tc['name']}")
                print(f"  With query:  {tc['args']}")
            tool_used = True

        elif msg_type == "ToolMessage":
            print(f"  Tool result: {str(msg.content)[:100]}...")

    if not tool_used:
        print("  No tools used — Claude answered from training data")

    final_answer = result["messages"][-1].content
    return final_answer

if __name__ == "__main__":

    # ── Test 1: Internal question → should use RAG ────────────────
    print("\n🔍 Test 1: Internal company question")
    r1 = run_agent(
        "What is TechCorp's refund policy?",
        thread_id="test-1"
    )
    print(f"\nFinal Answer:\n{r1}")

    # ── Test 2: External question → should use Tavily ─────────────
    print("\n\n🌐 Test 2: External public question")
    r2 = run_agent(
        "Who won the 2026 Spelling Bee?",
        thread_id="test-2"
    )
    print(f"\nFinal Answer:\n{r2}")

    # ── Test 3: Mixed question → should use both tools ────────────
    print("\n\n🔀 Test 3: Mixed question needing both tools")
    r3 = run_agent(
        "How does TechCorp's refund policy compare to industry standard?",
        thread_id="test-3"
    )
    print(f"\nFinal Answer:\n{r3}")

    # ── Test 4: Memory test ───────────────────────────────────────
    print("\n\n🧠 Test 4: Memory test across turns")
    run_agent(
        "I am on the Premium plan",
        thread_id="memory-test"
    )
    r4 = run_agent(
        "Based on my plan what are my support hours and SLA?",
        thread_id="memory-test"
    )
    print(f"\nFinal Answer:\n{r4}")

