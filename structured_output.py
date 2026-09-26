from typing import TypedDict, Annotated, Optional
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

#schema
class Person(TypedDict):
    name: Annotated[str, "The name of the person"]
    age: Annotated[int, "The age of the person"]
    email: Annotated[str, "The email address of the person"]

structured_model = model.with_structured_output(Person)
result = structured_model.invoke("Tell me about a fictional person named Rahul who is 25 years old and works as a software engineer.") 

print("RESPONSE TYPE:", type(result))
print("RESPONSE:", result)
print("NAME:", result["name"])
print("AGE:", result["age"])
print("EMAIL:", result["email"])