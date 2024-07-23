from langchain.memory.buffer import ConversationBufferMemory
from langchain.memory import ConversationSummaryMemory
from langchain.chains import LLMChain
from langchain.prompts import(
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)


from langchain_community.llms import OCIGenAI
import oci

#use the default authN method
llm = OCIGenAI(
    model_id = "cohere.command-light",
    service_endpoint = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id = "ocid1.compartment.oc1..aaaaaaaa345eyrbwg2ujmzjbhzuzi2szp64l3tx5ypeoqlnitxi7u6jlpsfa",
    model_kwargs={"max_tokens":100}
)

#Create memory to memorize chat with the llm
memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True)

summary_memory= ConversationSummaryMemory(llm=llm, memorykey="chat_history")

#Create conversation chain using llm, prompt and memory
conversation = LLMChain(llm=llm, prompt=prompt, verbose=True,memory=memory)

#invoke a chain
conversation.invoke({"question":"What is the capital of India"})

#print all messages into the memory
print(memory.chat_memory.messages)
print(summary_memory.chat.memory.messages)
print("Summary of conversation ->" + summary_memory.buffer)

