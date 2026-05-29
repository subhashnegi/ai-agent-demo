from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import textwrap

load_dotenv()
# Initialize the LLM
llm = ChatAnthropic(
    model="claude-sonnet-4-5",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=1024
)

#Prompt Engineering
SYSTEM_PROMPT= """You are a Research Assistant Agent. Your job is to help 
users find accurate, well-reasoned answers to their questions.

When answering:
1. Think step by step before giving your final answer (Chain of Thought)
2. Cite your reasoning clearly
3. If you're unsure, say so — never make up facts
4. Keep answers concise but complete

You have access to tools to search the web and query documents.
Always use tools when you need current or specific information."""

#Chat prompt template structures the converasation
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}")
])

# Chain
# The | operator chains components together: prompt → llm → parser
chain = prompt| llm | StrOutputParser()

# Structure output with Pydantic
class ResearchResponse(BaseModel):
    """Structured response from the research agent."""
    answer: str
    confidence: str #high med low
    needs_search: bool #does this needs web search


def research(question:str) -> str:
    """Run the research chain on a question."""
    return chain.invoke({"question": question})

if __name__== "__main__":
    response = research("Explain Chain of Thought prompting and why it improves LLM reasoning.")
    formatted_response = textwrap.fill(response, width=80)
    print(formatted_response)


