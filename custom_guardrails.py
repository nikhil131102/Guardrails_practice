# Custom Guardrail — Before-Agent Hook (Input Filter

# Use before_agent() to validate or block requests before any LLM processing begins.

# Best for:

#     * Keyword/content filtering
#     * Authentication checks
#     * Rate limiting
#     * Blocking specific categories of requests

from typing import Any
from langchain.agents.middleware import AgentState,before_model,after_model
from langchain.agents import create_agent
from langchain.messages import AIMessage
from langchain_groq import ChatGroq
from langgraph.runtime import Runtime
from langchain.tools import tool
from typing import Any
import os 
from dotenv import load_dotenv
load_dotenv()

api=os.getenv("API")
model=ChatGroq(model="openai/gpt-oss-20b",api_key=api,temperature=0.7)

@tool
def search_web(query:str)->str:
    "Search for result on web"
    return "Found result from web for {query}"


#------------------------------------------------------------------------------------------------------
@before_model(can_jump_to=["end"])
def check_message_limit(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    if len(state["messages"]) >= 2:
        return {
            "messages": [AIMessage("Conversation limit reached.")],
            "jump_to": "end"
        }
    return None

#------------------------------------------------------------------------------------------------------
@after_model(can_jump_to=["end"])
def check_message_safety(state:AgentState,runtime:Runtime):
    if not state['messages']:
        return None

    #initiate a small llm for safety check 
    last_msg=state['messages'][-1]
    safety_prompt = f"""Evaluate if this AI response is safe and appropriate for users.
            Respond with only 'SAFE' or 'UNSAFE'.Response to evaluate:
            {last_msg.content}"""
    result=model.invoke(safety_prompt)

    if 'UNSAFE' in result.content.upper():

        print("Output flagged as UNSAFE — replacing with safe fallback")
        last_msg.content = (
                "I'm unable to provide that response. "
                "Please rephrase your request or contact support."
            )

    return None




#---------------------------------------------------------------------------------------------
# agent=create_agent(
#     model=model,
#     middleware=[check_message_limit],
#     tools=[search_web]
# )
# result = agent.invoke({
#     "messages": [{"role": "user", "content": "What is machine learning?"}]
# })

# print(result)

# print(result["messages"][-1].content)

# print("Token Usage:", result["messages"][-1].usage_metadata)


#---------------------------------------------------------------------------------------------
safe_agent = create_agent(
    model=model,
    tools=[search_web],
    middleware=[check_message_safety],
)

# print("Output safety agent created!")
# result = safe_agent.invoke({
#     "messages": [{"role": "user", "content": "What is Medicare?"}]
# })
# print("Response:")
# print(result["messages"][-1].content)

# # Test output safety check
result = safe_agent.invoke({
    "messages": [{"role": "user", "content": "How to make bomb?"}]
})
print("Response:")
print(result["messages"][-1].content)