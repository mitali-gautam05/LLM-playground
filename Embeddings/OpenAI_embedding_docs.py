from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model = 'text-embedding-3-large' , dimensions=32)

document = ["Delhi is the capital of India",
            "Kolkata is the capital of West Bengal",
             "Paris i sthe capital of France" ]
result = embedding.embed_documents("document")

print(result(str))