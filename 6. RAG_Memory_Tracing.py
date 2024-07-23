from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationRetrievalChain
import chromadb
from langchain_community.vectorstores import Chroma
from chromadb.config import Settings

from langchain_community.llms import OCIGenAI
from langchain_community.embeddings import CohereEmbeddings

import os
from uuid import uuid4

unique_id = uuid4().hex[0:8]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"Test111 - {unique_id}"
os.environ["LANGCHAIN_ENDPOINT"] = "http://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "x"

#use the default authN method
llm = OCIGenAI(
    model_id = "cohere.command",
    service_endpoint = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id = "ocid1.compartment.oc1..aaaaaaaa345eyrbwg2ujmzjbhzuzi2szp64l3tx5ypeoqlnitxi7u6jlpsfa",
    model_kwargs={"max_tokens":400}
)

#Connect to a chromadb server, need to run chromadb server before connecting
client = chromadb.HttpClient(host="127.0.0.1")

#Create embeddings using cohere embedded light model 2.0
embeddings=OCIGenAIEmbeddings(
    model_id="cohere.embed-english-v3.0",
    service_endpoint= "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
compartment_id = "ocid1.compartment.oc1..aaaaaaaa345eyrbwg2ujmzjbhzuzi2szp64l3tx5ypeoqlnitxi7u6jlpsfa",
)

# Create a retriever that gets relevant docs (similar to a query)
db = Chroma(client=client, embedding_function=embeddings)

retv = db.as_retriever(search_type="similarity", search_kwargs={"k": 8})

# Explore doc similarity to query returned by printing doc metadata
doc = retv.get_relevant_documents('Tell us which module is most relevant to LLMs and Generative AI')

def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n"+d.page_content for i, d in enumerate(docs)]
        )

    )

pretty_print_docs(docs)

#create memory to remember chat messages
memory = ConversationBufferMemory(llm=llm, memory_key="chat history", return_messages=True, output_key='answer')

#Create a chain that uses llm, retriever and memory
qa = ConversationRetrievalChain.from_llm(llm, retriever=retv, memory=memory, return_source_documents=True)

response = qa.invoke({"question": "Tell us about Oracle Cloud Infrastructure AI Foundations course"})
print(memory.chat_memory.messages)

response = qa.invoke({"question": "Which module of the course is relevant to the LLMs and Transformers"})
print(memory.chat_memory.messages)

print(response)




