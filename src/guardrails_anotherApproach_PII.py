import re
import os
from dataclasses import dataclass
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from better_profanity import profanity
from dotenv import load_dotenv

load_dotenv()

# ── Data classes ──────────────────────────────────────────────────
# dataclass = Python's clean way to define data containers
# like a Java POJO — just stores data with typed fields

@dataclass
class InputValidationResult:
    """
    Result of validating user input.
    is_safe: True if input passes all checks
    blocked_reason: why it was blocked (None if safe)
    sanitised_input: cleaned version of input
    """
    is_safe:bool
    blocked_reason: str | None
    sanitised_input: str

@dataclass
class OutputValidationResult:
    """
   Result of validating agent output.
   is_safe: True if output passes all checks
   blocked_reason: why it was blocked (None if safe)
   safe_response: either original or fallback response
   confidence_score: 0.0 to 1.0
   """

    is_safe:bool
    blocked_reason: str | None
    safe_response: str
    confidence_score: float


# ── Constants ─────────────────────────────────────────────────────

# Prompt injection patterns to detect
# These are known attack phrases

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"forget\s+(everything|all|your\s+instructions)",
    r"(new|updated|override)\s+(system\s+prompt|instructions|rules)",
    r"you\s+are\s+now\s+",
    r"pretend\s+(you\s+are|to\s+be)",
    r"act\s+as\s+(if|though)",
    r"(dan|jailbreak|developer)\s+mode",
    r"print\s+(your\s+)?(system\s+prompt|instructions)",
    r"repeat\s+(your\s+)?(system\s+prompt|instructions)",
    r"i\s+am\s+from\s+anthropic",
    r"disregard\s+(your|all|previous)",
    r"bypass\s+(your|all|the)\s+(safety|security|guardrails)",
]

# PII patterns to detect and strip
PII_PATTERNS = {
    "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
    "CREDIT_CARD": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
    "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "PHONE": r"\b(\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b",
}

# Topics the agent is allowed to discuss
# Anything outside this list is off-topic
ALLOWED_TOPICS = [
    "refund", "return", "money back", "cancel",
    "subscription", "plan", "pricing", "cost", "price",
    "support", "help", "hours", "contact", "ticket",
    "privacy", "data", "gdpr", "security",
    "api", "integration", "technical",
    "account", "billing", "payment", "invoice",
    "premium", "professional", "enterprise", "basic",
    "spelling bee", "weather", "news",  # allow general queries for demo
]

# Fallback responses for different failure modes
FALLBACK_RESPONSES = {
    "injection": (
        "I can only help with TechCorp support questions. "
        "How can I assist you with your account today?"
    ),
    "off_topic": (
        "I am a TechCorp customer support assistant. "
        "I can help with refunds, subscriptions, technical support, "
        "and account questions. What can I help you with?"
    ),
    "hallucination": (
        "I want to make sure I give you accurate information. "
        "Let me connect you with our support team who can "
        "verify the details for you."
    ),
    "too_short": (
        "I was unable to generate a complete response. "
        "Please try rephrasing your question or contact "
        "support at support@techcorp.com."
    ),
    "pii_detected": (
        "For your security please do not share sensitive personal "
        "information like social security numbers or credit card "
        "details in chat. How can I help you with your account?"
    ),
}

# ── Input Guardrails ──────────────────────────────────────────────

def check_prompt_injection(text:str)-> bool:
    """
    Checks if input contains known prompt injection patterns.
    Returns True if injection detected (unsafe).

    Why regex not LLM:
    → faster — no API call needed
    → deterministic — always catches known patterns
    → cheaper — zero cost
    → LLM-based check added on top for novel attacks
    """

    text_lower = text.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern,text_lower):
            return True
    return False

def check_pii(text:str) -> tuple[bool,str]:
    """
    Checks if input contains PII patterns.
    Returns (pii_found, sanitised_text).

    We sanitise rather than block entirely —
    user may not realise they are sharing PII.
    Better to strip it and continue than to reject.
    """

    sanitised = text
    pii_found = False

    for pii_type , pattern in PII_PATTERNS.items():
        if re.search(pattern,sanitised):
            pii_found = True
            sanitised = re.sub(
                pattern,f"[{pii_type}_REDACTED]",
                sanitised
            )
    return pii_found, sanitised


def check_toxicity(text:str)-> bool:
    """
    Checks if input contains toxic or profane language.
    Returns True if toxic (unsafe).

    Uses better-profanity library —
    runs locally, no API key needed, fast.
    """
    return profanity.contains_profanity(text)

def check_topic_relevance(text:str)-> bool:
    """
   Checks if input is relevant to our support domain.
   Returns True if relevant (safe to proceed).

   Simple keyword matching —
   production would use embedding similarity
   to handle semantic variations.
   """
    text_lower = text.lower()
    return any(topic in text_lower for topic in ALLOWED_TOPICS)

def validate_input(user_input:str) -> InputValidationResult:
    """
    Master input validation function.
    Runs all checks in priority order.
    Returns InputValidationResult with decision.

    Priority order:
    1. Injection (most dangerous — block immediately)
    2. Toxicity (abusive — block immediately)
    3. PII (sanitise and continue)
    4. Topic relevance (redirect if off-topic)
    """
    # Check 1 — Prompt injection (highest priority)
    if check_prompt_injection(user_input):
        return InputValidationResult(
            is_safe =False,
            blocked_reason ="prompt_injection",
            sanitised_input= user_input
        )

    # Check 2 — Toxicity
    if check_toxicity(user_input):
        return InputValidationResult(
            is_safe =False,
            blocked_reason ="toxic_content",
            sanitised_input= user_input
        )

    # Check 3 — PII (sanitise but allow through)
    pii_found, sanitised = check_pii(user_input)
    if pii_found:
        # Log PII detection but sanitise and continue
        #print(f"  ⚠️  PII detected and redacted in input")
        return InputValidationResult(
            is_safe=False,
            blocked_reason="pii_detected",
            sanitised_input=sanitised
        )

    # Check 4 — Topic relevance
    if not check_topic_relevance(user_input):
        return InputValidationResult(
            is_safe=False,
            blocked_reason="off_topic",
            sanitised_input=user_input
        )


    # All checks passed
    return InputValidationResult(
        is_safe=True,
        blocked_reason=None,
        sanitised_input=user_input
    )


# ── Output Guardrails ─────────────────────────────────────────────

def check_answer_completeness(answer: str) -> bool:
    """
    Checks if answer is substantive enough to be useful.
    Returns True if complete (safe).

    Too-short answers indicate:
    → agent failed to retrieve relevant information
    → LLM gave up without answering
    → tool failure caused incomplete response
    """
    # Minimum 30 words for a meaningful support answer
    word_count = len(answer.split())
    return word_count >= 30

def check_uncertainty_markers(answer: str) -> tuple[bool, str]:
    """
    Detects when agent is uncertain but presenting as fact.
    Returns (has_uncertainty, annotated_answer).

    Uncertain answers are acceptable — but must be flagged
    so user knows to verify. Confident wrong answers are dangerous.
    """
    uncertainty_phrases = [
        "i believe", "i think", "i'm not sure",
        "might be", "could be", "probably",
        "i'm not certain", "i cannot confirm",
        "you may want to verify"
    ]

    answer_lower = answer.lower()
    has_uncertainty = any(
        phrase in answer_lower
        for phrase in uncertainty_phrases
    )
    if has_uncertainty:
        # Add disclaimer rather than blocking
        annotated = (
                answer +
                "\n\n⚠️ Note: Please verify this with our support "
                "team at support@techcorp.com for confirmation."
        )
        return True, annotated

    return False, answer

def check_hallucination_llm(
        question: str,
        answer: str,
        source_chunks: list[str]
) -> tuple[bool, float]:
    """
    Uses LLM-as-judge to check if answer is grounded
    in source documents.

    Returns (is_grounded, confidence_score).

    Uses Haiku — cheap fast model — affordable to run
    on every query without significant cost impact.

    Why LLM-as-judge:
    → more nuanced than keyword matching
    → understands paraphrasing and synonyms
    → catches semantic hallucination not just word mismatch
    """

    if not source_chunks:
        # No sources to check against — skip grounding check
        return True, 0.8

    eval_llm = ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=200
    )

    sources_text = "\n\n".join([
        f"Source {i+1}: {chunk[:300]}"
        for i, chunk in enumerate(source_chunks[:3])
    ])

    eval_prompt = f"""You are a fact checker for a customer support system.

    Question: {question}

    Source documents provided to the agent:
    {sources_text}

    Agent answer: {answer[:500]}

    Is the agent's answer grounded in the source documents?
    Reply with ONLY one of these exact responses:
    GROUNDED:0.95
    GROUNDED:0.80
    UNCERTAIN:0.60
    NOT_GROUNDED:0.30
    NOT_GROUNDED:0.10

    Format: STATUS:SCORE (no other text)"""

    try:
        response = eval_llm.invoke([
            HumanMessage(content=eval_prompt)
        ])
        result = response.content.strip()

        # Parse response
        if ":" in result:
            status, score_str = result.split(":", 1)
            score = float(score_str)
            is_grounded = "NOT_GROUNDED" not in status
            return is_grounded, score
    except Exception as e:
        print(f"  ⚠️  Hallucination check error: {e}")

    # Default to safe if check fails
    return True, 0.7


def validate_output(
        question: str,
        answer: str,
        source_chunks: list[str] = None
) -> OutputValidationResult:
    """
    Master output validation function.
    Runs all output checks in priority order.

    source_chunks: the RAG chunks used to generate answer
                   None for web search answers
    """
    if source_chunks is None:
        source_chunks = []

    # Check 1 — Completeness
    if not check_answer_completeness(answer):
        return OutputValidationResult(
            is_safe=False,
            blocked_reason="too_short",
            safe_response=FALLBACK_RESPONSES["too_short"],
            confidence_score=0.0
        )

    # Check 2 — Uncertainty markers (annotate not block)
    has_uncertainty, annotated_answer = check_uncertainty_markers(answer)
    working_answer = annotated_answer

    # Check 3 — Hallucination detection (only for RAG answers)
    if source_chunks:
        is_grounded, confidence = check_hallucination_llm(
            question, answer, source_chunks
        )
        if not is_grounded:
            return OutputValidationResult(
                is_safe=False,
                blocked_reason="hallucination_detected",
                safe_response=FALLBACK_RESPONSES["hallucination"],
                confidence_score=confidence
            )
    else:
        confidence = 0.85  # default for web search answers

    # All checks passed
    return OutputValidationResult(
        is_safe=True,
        blocked_reason=None,
        safe_response=working_answer,
        confidence_score=confidence
    )

# ── Unified guardrail wrapper ─────────────────────────────────────

def run_with_guardrails(
        question: str,
        agent_func,
        thread_id: str = "default",
        source_chunks: list[str] = None
) -> dict:
    """
    Wraps any agent function with input and output guardrails.

    Usage:
    result = run_with_guardrails(
        question="What is the refund policy?",
        agent_func=run_agent,
        thread_id="CUST_001"
    )

    Returns dict with:
    answer: the safe response
    blocked: True if blocked at any point
    block_reason: why it was blocked
    confidence: output confidence score
    """

    print(f"\n{'='*50}")
    print(f"Question: {question}")
    print(f"{'='*50}")

    # ── INPUT GUARDRAILS ──────────────────────────────────────────
    print("\n[1/3] Input validation...")
    input_result = validate_input(question)
    if not input_result.is_safe:
        reason = input_result.blocked_reason

        # PII is special — sanitise and continue
        if reason == "pii_detected":
            print(f"  ⚠️  PII detected and redacted")
            print(f"  Original:  {question[:80]}")
            print(f"  Sanitised: {input_result.sanitised_input[:80]}")
            print(f"  Continuing with sanitised input...")

            answer = agent_func(
                input_result.sanitised_input,
                thread_id
            )

            output_result = validate_output(
                question, answer, source_chunks or []
            )

            if not output_result.is_safe:
                return {
                    "answer": output_result.safe_response,
                    "blocked": True,
                    "block_reason": output_result.blocked_reason,
                    "confidence": output_result.confidence_score
                }

            safe_answer = (
                    "⚠️ Note: Sensitive information was removed from "
                    "your message for your security.\n\n" +
                    output_result.safe_response
            )

            return {

                "answer": safe_answer,
                "blocked": False,
                "block_reason": "pii_redacted_and_continued",
                "confidence": output_result.confidence_score
            }

        # All other violations — block immediately
        print(f"  ❌ Input blocked: {reason}")
        return {
            "answer": FALLBACK_RESPONSES.get(reason, FALLBACK_RESPONSES["injection"]),
            "blocked": True,
            "block_reason": reason,
            "confidence": 0.0
        }
    print(f"  ✅ Input passed all checks")

    # ── AGENT RUNS ────────────────────────────────────────────────
    print("\n[2/3] Running agent...")
    answer = agent_func(
        input_result.sanitised_input,
        thread_id
    )
    print(f"  ✅ Agent completed")

    # ── OUTPUT GUARDRAILS ─────────────────────────────────────────
    print("\n[3/3] Output validation...")
    output_result = validate_output(
        question,
        answer,
        source_chunks
    )

    if not output_result.is_safe:
        print(f"  ❌ Output blocked: {output_result.blocked_reason}")
        return {
            "answer": output_result.safe_response,
            "blocked": True,
            "block_reason": output_result.blocked_reason,
            "confidence": output_result.confidence_score
        }

    print(f"  ✅ Output passed (confidence: {output_result.confidence_score:.0%})")

    return {
        "answer": output_result.safe_response,
        "blocked": False,
        "block_reason": None,
        "confidence": output_result.confidence_score
    }

