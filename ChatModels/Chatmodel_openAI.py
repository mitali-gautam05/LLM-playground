from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model = 'gpt-4' , temperature=1.5 , max_completion_tokens=13333)
# tempearture is for the response like 1.5 willbe more chaotic , 0.5 will be serious good for coding 

result = model.invoke("What is the capital of India?")

print(result.content)