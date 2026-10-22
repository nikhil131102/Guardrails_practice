#WHAT ARE GUARDRAILS?
#  In AI/LLM systems, guardrails are validation and control layers around model inputs, tools, and outputs. They enforce policies such as “no PII,” “always valid JSON,” or “never execute arbitrary code.”​
# Practically, they are rules, validators, or secondary models that inspect data before and after the LLM call, approving, fixing, or blocking it as needed.


from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware
from langchain_groq import ChatGroq
import os 
from dotenv import load_dotenv
load_dotenv()
api=os.getenv("API")
model=ChatGroq(model="openai/gpt-oss-20b",api_key=api,temperature=0.7)

agent=create_agent(model=model,
                   middleware=[PIIMiddleware("email",strategy='redact',apply_to_input=True),
                               PIIMiddleware("credit_card",strategy='mask',apply_to_input=True),
                               PIIMiddleware("url",strategy='redact',apply_to_input=True),
                                PIIMiddleware("ip",strategy='hash',apply_to_input=True),
                                PIIMiddleware("mac_address",strategy='hash',apply_to_input=True)
                            ]
                               


                   )

result=agent.invoke({
    "messages":[{"role":"user",
                 "content":"My name is nikhil and my mail is nikhil131102@gmail.com and 5105-1051-0510-5100 along with www.google.com"}]
})

print(result['messages'][0].content)