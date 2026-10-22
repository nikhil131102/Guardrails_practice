# Built-in Guardrail — Human-in-the-Loop Middleware
# Pauses agent execution before sensitive operations and waits for human approval.

# Best for:

# Financial transactions
# Sending emails to external parties
# Deleting production data
# Any operation with significant business impact
# Key requirement: A checkpointer for state persistence across interrupts.

from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware,HumanInTheLoopMiddleware
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool
from langgraph.types import Command

import os 
from dotenv import load_dotenv
load_dotenv()

api=os.getenv("API")
model=ChatGroq(model="openai/gpt-oss-20b",api_key=api,temperature=0.7)

@tool
def search_web(query:str)->str:
    "Search for result on web"
    return "Found result from web for {query}"

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to a recipient."""
    return f"Email sent to {to} with subject: {subject}"

@tool
def delete_records(table: str, condition: str) -> str:
    """Delete records from the database."""
    return f"Deleted records from {table} where {condition}"

#Create an HITL agent

hitl=create_agent(model=model,tools=[search_web,delete_records,send_email],
            middleware=[HumanInTheLoopMiddleware(
                interrupt_on={"send_email":True,
                              "delete_records":True,
                              "search_web":False}

                  )],checkpointer=InMemorySaver() #for state persistant
                  )





config = {"configurable": {"thread_id": "session_001"}}
result=hitl.invoke({
    "messages": [{"role": "user", "content": "Send an email to team@company.com about the Q4 results"}]
},config=config)


print("=== Agent paused — awaiting human approval ===")
#print(result)



###################-------------------------------------------#####################################

approved_result = hitl.invoke(
    Command(resume={"decisions": [{"type": "approve"}]}),
    config=config   # Same thread_id resumes the paused session
)

print("=== Approved! Final response ===")
print(approved_result["messages"][-1].content)

###################-------------------------------------------#####################################

# Alternative — Human REJECTS

config2 = {"configurable": {"thread_id": "session_002"}}

hitl.invoke(
    {"messages": [{"role": "user", "content": "Delete all records from the users table where active=false"}]},
    config=config2
)

print("=== Agent paused — awaiting human approval ===")
#print(result)

###################-------------------------------------------#####################################

rejected_result = hitl.invoke(
    Command(resume={"decisions": [{"type": "reject", "reason": "Too risky, needs DBA review"}]}),
    config=config2
)

print("=== Rejected! Final response ===")
print(rejected_result["messages"][-1].content)