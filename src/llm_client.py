import os
import textwrap
from anthropic import Anthropic
from dotenv import load_dotenv

#load .env file
load_dotenv()

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def ask(question: str) -> str:
    """Send a question to Claude and return the response"""
    message = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        messages=[
            {"role":"user","content":question},
        ]

    )
    return message.content[0].text

def ask_streaming(question: str) -> None:
    """Send a question and stream the response word by word."""

    print("Assistant: ", end= "", flush=True)

    with client.messages.stream(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        messages=[
            {"role":"user","content": question}
        ]
    ) as stream:
        for text in stream.text_stream:
            print(text,end="",flush=True)

    print()




if __name__== "__main__":
    #Test Basic call
    #response = ask("What is RAG pipeline in AI? Explain in 3 sentences.")
    #print("\n--- FORMATTED OUTPUT ---")
    #formatted_response = textwrap.fill(response, width=80)
    #print(formatted_response)

    #Test Streaming
    print("Streaming response:")
    ask_streaming("What is an AI agent? Explain in 3 sentences.")
