from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Flash-0731",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template="Greet this person in 5 languages. The name of the person is {name}",
    input_variables=["name"]
)

chain = prompt | model

result = chain.invoke({"name": "Mitali"})

print(result.content)