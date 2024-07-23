from langchain.memory.buffer import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import OCIGenAI
import oci

#use the default authN method
llm = OCIGenAI(
    model_id = "cohere.command-light",
    service_endpoint = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id = "x",
    model_kwargs={"max_tokens":100}
)

#Create history w/ a key chat msg

#Streamlitchatmessagehistory will store msg in streamlit sesh at the specified keys.
#A given Streamlitchatmessagehistory will not be persisted or share across user sesh.

history = StreamlitChatMessageHistory(key="chat_messages")

#Create memory object
memory=ConversationBufferMemory(chat_memory=history)


#Create template and prompt to accept a question
template="""You are an AI chatbot having a converstion with a human.
Human: {user_input}
AI: """
prompt = PromptTemplate(input_variables=["human_input"],template=template)

#Create chain object
llm_chain=LLMChain(llm=llm,prompt=prompt,memory=memory)

#Use streamlit to print all msg

import streamlit as st

st.title("Welcome to the AI Chatbot GAI")
for msg in history_messages:
    st.chat_message(msg.type).write(msg.content)

if x:= st.chat_input():
    st.chat_message("human").write(x)

    #new msg added to streamlitchatmsghistory when chain is called
    response = llm_chain.run(x)
    st.chat_message("ai").write(response)
