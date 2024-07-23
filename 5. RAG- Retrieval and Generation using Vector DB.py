from langchain.chains import RetrievalQA
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_community.llms import OCIGenAI
from langchain_community.embeddings import CohereEmbeddings


#use the default authN method
llm = OCIGenAI(
    model_id = "cohere.command-light",
    service_endpoint = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id = "ocid1.compartment.oc1..aaaaaaaa345eyrbwg2ujmzjbhzuzi2szp64l3tx5ypeoqlnitxi7u6jlpsfa",
    model_kwargs={"max_tokens":100}
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

retv = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Explore doc similarity to query returned by printing doc metadata
doc = retv.get_relevant_documents('Tell us which module is most relevant to LLMs and Generative AI')

def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n"+d.page_content for i, d in enumerate(docs)]
        )

    )

pretty_print_docs(docs)

for docs in docs:
    print(doc.metadata)

#Create retrieval chain that takes llm, retriever objects and invoke to to get a response
chain = RetrievalQA.from_chain_type(llm=llm,retriever=retv,return_source_documents=True)

response = chain.invoke("Tell us which module is most relevant to LLMs and Generative AI")

print(response)




