from typing import Optional
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

class User(BaseModel):

    id: int = Field(
        ...,
        description="The unique identifier for the user"
    )

    name: str = Field(
        ...,
        description="The name of the user",
        min_length=1,
        max_length=100
    )

    age: int = Field(
        ...,
        description="The age of the user",
        ge=0,
        le=120
    )

    is_active: bool = Field(
        ...,
        description="Indicates whether the user is active"
    )

    occupation: Optional[str] = Field(
        None,
        description="The occupation of the user",
        min_length=1,
        max_length=100
    )

    mobile_number: Optional[str] = Field(
        None,
        description="The mobile number of the user",
        pattern=r'^\+?[1-9]\d{1,14}$'
    )


user = User(
    id=1,
    name="John Doe",
    age=30,
    is_active=True,
    occupation="Software Engineer",
    mobile_number="+1234567890"
)

structured_model = model.with_structured_output(User)

result = structured_model.invoke("Tell me about a fictional user named John Doe who is 30 years old, is active, works as a software engineer, and has a mobile number +1234567890.")


print("TYPE:", type(result))
print("RESPONSE:", result)

print("NAME:", result.name)
print("AGE:", result.age)
print("OCCUPATION:", result.occupation)

print("DICTIONARY:", result.model_dump())
