from langchain_community.embeddings import OCIGenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

from langchain_community.embeddings import CohereEmbeddings

pdf_loader = PyPDFDirectoryLoader(".\pdf-docs")

loaders= [pdf_loader]

documents=[]
for loader in loaders:
    documents.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=100)
all_documents=text_splitter.split_documents(documents)

print(f"total numbers of documents: {len(all_documents)}")

#use the default authN method
llm = OCIGenAI(
    model_id = "cohere.embed-english-v3.0",
    service_endpoint = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id = "x",
    model_kwargs={"truncate":True}
)

#OCIGenAIEmbeddings accepts only 96 docs (1) run -> input doc into batches

#Set batch size
batch_size=96

#Calculate num of batches
num_batches= len(all_documents) // batch_size + (len(all_documents) % batch_size > 0)

db = Chroma(embedding_function=embeddings, persist_directory="./chromadb")
retv = db.as_retriever()

#Iterate over batches
for batch_num in range(num_batches):
    #Calculate start-end indices for the current batch
    start_index = batch_num * batch_size
    end_index = (batch_num+1) * batch_size
    #Extract docs for current batch
    batch_documents= all_documents[start_index:end_index]
    #Process each document <Code Here>
    retv.add_documents(batch_documents)
    print(start_index, end_index)

#Persist the collection
db.persist()


