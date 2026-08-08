from langchain_core.messages import SystemMessage , HumanMessage , AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint 
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Flash-0731",
    task="text-generation"
)

model = ChatHuggingFace(llm = llm)

messages = [ SystemMessage(content = 'you are helpful assistant'),
             HumanMessage(content = 'tell me about langchain')
            ]

result = model.invoke(messages)

messages.append(AIMessage(content = result.content))

print(messages)