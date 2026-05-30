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

#this one is not able to search latest information so it has modification rate low for such result.

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
    response=structured_chain.invoke({"question": question})

    # Override: if LLM admits uncertainty, force needs_search
    uncertainty_phrases = [
        "i cannot", "i don't know", "my knowledge cutoff","i don't have information",
        "future event", "i'm not sure", "as of my knowledge"
    ]
    answer_lower = response.answer.lower()
    if any(phrase in answer_lower for phrase in uncertainty_phrases):
        # Force needs_search to True
        return ResearchResponse(
            answer=response.answer,
            confidence="low",
            needs_search=True  # override LLM's wrong decision
        )

    return response

if __name__== "__main__":
    response = research("Who won 2026 Spell Bee?.")
    print(f"Answer:  {response.answer}")
    print(f"Answer:  {response.confidence}")
    print(f"Answer:  {response.needs_search}")

