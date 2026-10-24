# Real-World Use Case — Healthcare Chatbot
# A healthcare chatbot that:

# Blocks off-topic or harmful requests
# Redacts patient PII (emails, credit card numbers)
# Requires human approval before booking appointments
# Validates that outputs are medically appropriate

from langchain.tools import tool
from langgraph.types import Command
from langgraph.runtime import Runtime
from langchain_groq import ChatGroq
from langchain.agents import create_agent,AgentState
from langchain.agents.middleware import PIIMiddleware,before_model,after_model,HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver

import os
from dotenv import load_dotenv

load_dotenv()

api=os.getenv("API")

model=ChatGroq(model='openai/gpt-oss-20b',temperature=0.7,api_key=api)

@tool
def booking_appoint(patient:str)->str:
    """Book an appointment on behalf of patient"""
    return "Booking Done for {patient}"

@after_model(can_jump_to=['end'])
def safe_response(state:AgentState,runtime:Runtime):
    DISCLAIMER="*This is general health information, not medical advice. Please consult a qualified healthcare professional"

    last_msg=state['messages'][-1]
    prompt=f""" You are an healthcare bot and your only work is to check the response provided to you.If the response is medically appropriate then return 'SAFE' else return 'UNSAFE'. response is {last_msg}"""

    result=model.invoke(prompt)

    if 'UNSAFE' in result.content.upper():
        print("Unsafe response")
        last_msg.content="Cannot provide output"

    if "medical advice" not in last_msg.content.lower():
            last_msg.content +=DISCLAIMER
        
    return None



bot=create_agent(model=model,tools=[booking_appoint],
                 middleware=[PIIMiddleware('email',strategy='redact'),
                                         PIIMiddleware('credit_card',strategy='redact'),
                                         HumanInTheLoopMiddleware({'booking_appoint':True}),
                                         safe_response
                                         ],
                                         
                                         checkpointer=InMemorySaver(),
                                         system_prompt=("You are a helpful healthcare assistant. "
        "You can search for symptoms, medication information, and help book appointments. "
        "Always be empathetic and remind users to consult a doctor for diagnosis."))


# Test 1: Safe medical query
config_t1 = {"configurable": {"thread_id": "healthcare_session_t1"}}

result = bot.invoke(
    {"messages": [{"role": "user", "content": "What are symptoms of Type 2 Diabetes?"}]},
    config=config_t1
)

print(result["messages"][-1].content)