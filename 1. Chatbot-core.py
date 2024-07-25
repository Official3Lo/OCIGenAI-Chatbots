from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

from langchain_community.llms import OCIGenAI
import oci

#OCI Gen AI llm
endpoint = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"

#use the default authN method
llm = OCIGenAI(
    model_id = "cohere.command-light",
    service_endpoint = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id = "x",
    model_kwargs={"max_tokens":100}
)

#invoke llm with a fixed text input
response = llm.invoke("Tell me one fact about space", temperature=0.7)
print("Case1 Response ->" + response)

#^^^Before all this, setup all mods at Oracle Cloud before this works:^^^
#(2)DedicatedAICluster[Hosting/Fine-tuning]
#Custom Model using any of the DedAIClust
#Endpoint for response

#Used String Prompt to accept text input. Here we create a template and declare input variable <User Input>
#string prompt

template = """You are a chatbot having a conversation with a human.
Human: {user_input} + {city}
:"""

#Created a prompt using the template

prompt = PromptTemplate(input_variables=["user_input","city"], template=template)

prompt_val=prompt.invoke({"user_input":"Tell us in an exciting tone about", "city":"Las Vegas"})
print("Prompt string is ->")
print(prompt_val.to_string())

#Declared chain that begins the prompt, next llm and final output of parser
chain = prompt | llm

#Invoked chain and provide input question
response = chain.invoke({"user_input":"Tell is in an exciting tone about", "city":"Las Vegas"})

#print prompt and response from llm
print("Case2 Response ->"+response)

#Used Chat Prompt to accept text input. Created a chat template and used HumanMessage and SystemMessage

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a chatbot that explains in steps"),
        ("ai", "I shall explain in steps"),
        ("human","{input}"),
    ]
)

chain= prompt | llm
response = chain.invoke({"input": "What's the New York culture like?"})
print("Case3 Response ->" + response)

