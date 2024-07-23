import os
from uuid import uuid4
from langsmith import Client

unique_id = uuid4().hex[0:8]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"Test111 - {unique_id}"
os.environ["LANGCHAIN_ENDPOINT"] = "http://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "x"

#Create dataset for evaluation
dataset_inputs = [
    "Tell us about Oracle Cloud Infrastructure AI Foundation Course and Certification",
    "Tell us which module in this course is relevant to Deep Learning",
    "Tell us about which module is relevant to LLMs and Transformers",
    "Tell me about the instructors of this course",
    #...
]

#Outputs provided to the evaluator for comparison
#Optional, but recommended
datasets_outputs = [
    {"must_mention": ["AI","LLM"]},
    {"must_mention": ["CNN","Neural Network"]},
    {"must_mention": ["Module 5", "Transformer", "LLM"]},
    {"must_mention": ["Hemant", "Himanshu", "Nick"]}
]

client = Client()
dataset_name = 'AIFoundationsDS-111'

#Storing inputs in a dataset lets us
#run chains and LLMs over a shared set of examples
dataset = client.create_dataset(
    dataset_name=dataset_name,
    description = 'AI Foundations QA'
)
client.create_example(
    inputs = [{"question": q} for q in dataset_inputs],
    outputs = datasets_outputs,
    dataset_id = dataset.id,
)


