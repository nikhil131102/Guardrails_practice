# Stack multiple guardrails in the middleware=[] array. They execute in order, building layered protection.

# User Input
#     ↓
# [Layer 1] ContentFilterMiddleware    ← Deterministic input filter
#     ↓
# [Layer 2] PIIMiddleware (input)      ← PII redaction on input
#     ↓
# [Layer 3] HumanInTheLoopMiddleware   ← Approval for sensitive tools
#     ↓
# [Layer 4] PIIMiddleware (output)     ← PII redaction on output
#     ↓
# [Layer 5] SafetyGuardrailMiddleware  ← Model-based output safety
#     ↓
# User Response


from typing import Any
from langchain.agents.middleware import AgentState,before_model,after_model,PIIMiddleware,HumanInTheLoopMiddleware
from langchain.agents import create_agent
from langchain.messages import AIMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime
from langgraph.types import Command
from langchain.tools import tool
from typing import Any
import os 
from dotenv import load_dotenv
load_dotenv()


api=os.getenv("API")
model=ChatGroq(model="openai/gpt-oss-20b",api_key=api,temperature=0.7)


@tool
def search_tool(query: str) -> str:
    """Search for information."""
    return f"Search results: {query}"

@tool
def send_email_tool(to: str, body: str) -> str:
    """Send an email."""
    return f"Email sent to {to}"

@after_model(can_jump_to=['end'])
def safe_response(state:AgentState,runtime:Runtime):
    latest_res=state['messages'][-1]
    safe_prompt=f"""You are a helpful agent who helps to guide user weather the ai response is safe or not, answer only 'UNSAFE' or 'SAFE'. Here is the response{latest_res}"""

    result=model.invoke(safe_prompt)
    #print(result)

    if 'UNSAFE' in result.content.upper():
        print('UNSAFE') 
        state['messages'][-1]="Cannot process the request.!"
    return None




# Full layered guardrail stack

prod_agent=create_agent(model=model,
                        tools=[search_tool,send_email_tool],
                        middleware=[
    #PIIMiddleware('email',strategy='redact',),
    HumanInTheLoopMiddleware(interrupt_on={'send_email_tool':True,'search_tool':False}),
    PIIMiddleware('credit_card',strategy='mask'),
    safe_response
],checkpointer=InMemorySaver())


config = {"configurable": {"thread_id": "test_001"}}

result = prod_agent.invoke(
    {"messages": [{"role": "user", "content": "Send an email to team@company.com about the Q4 results "}]},
    config=config
)
print(result["messages"][-1].content)


approved_result = prod_agent.invoke(
    Command(resume={"decisions": [{"type": "approve"}]}),
    config=config   # Same thread_id resumes the paused session
)

print("=== Approved! Final response ===")
print(approved_result["messages"][-1].content)