# --- Model-based approach ---
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware
from langchain_groq import ChatGroq
import os 
from dotenv import load_dotenv
load_dotenv()

api=os.getenv("API")
model=ChatGroq(model="openai/gpt-oss-20b",api_key=api,temperature=0.7)


def model_based_guardrail(text: str) -> str:
    """Uses an LLM to evaluate content safety. Returns SAFE or UNSAFE."""

    prompt = f"""Is the following user input safe to process?
Reply with only 'SAFE' or 'UNSAFE'.

Input: {text}"""
    result = model.invoke([{"role": "user", "content": prompt}])
    return result.content.strip()

print("=== Model-Based Guardrail Demo ===")


test_inputs=["I want to take part in hackathon.",
             "I want to hack my mobile phone.",
             "how to take access of someone else's phone?",
             "how malware spreads?"]


for inp in test_inputs:
    verdict = model_based_guardrail(inp)
    status = "UNSAFE" if "UNSAFE" in verdict else "SAFE"
    print(f"{status}: {inp}")