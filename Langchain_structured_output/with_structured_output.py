from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from typing import TypedDict ,Annotated, Optional, Literal

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Flash-0731",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

# schema
class Review(TypedDict):

    key_themes: Annotated[list[str], "Write down all the key themes discussed in the review in a list"]
    summary: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[Literal["pos", "neg" , "neutral"], "Return sentiment of the review either negative, positive or neutral"]
    pros: Annotated[Optional[list[str]], "Write down all the pros inside a list"]
    cons: Annotated[Optional[list[str]], "Write down all the cons inside a list"]
    name: Annotated[Optional[str], "Write the name of the reviewer"]



structured_model = model.with_structured_output(Review)


result = structured_model.invoke(
    """Analyze this review:

I recently upgraded to the Samsung Galaxy S24 Ultra.
The processor is extremely fast and the camera is excellent.
The battery life is great, but the phone is heavy and expensive.

Return the result according to the provided schema.

Review by Mitali Gautam
"""
)

print(result)
