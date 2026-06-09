import os
from langsmith import traceable
from dotenv import load_dotenv

load_dotenv()

from rag_agent import run_agent

@traceable(name="customer_support_query")
def handle_customer_query(query: str, customer_id: str) -> dict:
    """
    Simulates a real customer support query.
    Fully traced in LangSmith automatically.

    @traceable:
    → wraps this function
    → sends trace to LangSmith
    → shows up as named span in dashboard
    → inputs and outputs both logged
    """
    thread_id = f"customer_{customer_id}"
    answer = run_agent(query, thread_id)

    return {
        "customer_id": customer_id,
        "query": query,
        "answer": answer
    }

@traceable(name="evaluate_answer")
def evaluate_answer(query: str, answer: str, expected: str) -> dict:
    """
    Simple evaluation — did agent get the right answer?
    Key fact assertion — not exact match.
    This is non-deterministic AI validation.
    """
    passed = expected.lower() in answer.lower()

    return {
        "passed": passed,
        "expected": expected,
        "score": 1.0 if passed else 0.0
    }

if __name__ == "__main__":
    print("Running traced customer support queries...")
    print("Watch https://smith.langchain.com for live traces\n")

    test_cases = [
        {
            "customer_id": "CUST_001",
            "query": "What is your refund policy?",
            "expected": "30 days"
        },
        {
            "customer_id": "CUST_002",
            "query": "How much is the Premium plan?",
            "expected": "99.99"
        },
        {
            "customer_id": "CUST_003",
            "query": "Who won the 2026 Spelling Bee?",
            "expected": "Shrey Parikh"
        }
    ]

    results = []
    for test in test_cases:
        print(f"Processing {test['customer_id']}...")

        result = handle_customer_query(
            test["query"],
            test["customer_id"]
        )

        evaluation = evaluate_answer(
            test["query"],
            result["answer"],
            test["expected"]
        )

        results.append(evaluation)

        print(f"  Query:    {test['query']}")
        print(f"  Expected: {test['expected']}")
        print(f"  Passed:   {'✅' if evaluation['passed'] else '❌'}")
        print()

    passed_count = sum(1 for r in results if r["passed"])
    print("="*50)
    print(f"Results: {passed_count}/{len(results)} passed")
    print(f"Success rate: {passed_count/len(results):.0%}")
    print(f"\nView traces at https://smith.langchain.com")
    print("="*50)