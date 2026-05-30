from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatAnthropic(
    model="claude-sonnet-4-5",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=1024
)

#SYSTEM_PROMPT = """You are a Research Assistant Agent. Your job is to help
#users find accurate, well-reasoned answers to their questions.

#When answering:
#1. Think step by step before giving your final answer (Chain of Thought)
#2. Cite your reasoning clearly
#3. If you are unsure, say so — never make up facts
#4. Keep answers concise but complete"""

SYSTEM_PROMPT = """You are a Research Assistant.

CRITICAL RULES:
- Your knowledge cuts off at early 2025
- For ANY event after 2025, set needs_search to True
- For ANY sports results, winners, elections, set needs_search to True
- Never say you cannot answer — instead set needs_search to True
  and the system will search for you"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}")
])

# Structure output with Pydantic
class ResearchResponse(BaseModel):
    """Structured response from the research agent."""
    answer: str
    confidence: str #high med low
    needs_search: bool #does this needs web search


structured_llm = llm.with_structured_output(ResearchResponse)
structured_chain = prompt | structured_llm

def research(question:str) -> ResearchResponse:
    """Run the research chain on a question."""
    return structured_chain.invoke({"question": question})

if __name__== "__main__":
    response = research("Who won 2026 Spell Bee?.")
    print(f"Answer:  {response.answer}")
    print(f"Answer:  {response.confidence}")
    print(f"Answer:  {response.needs_search}")

