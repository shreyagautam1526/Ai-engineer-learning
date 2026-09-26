from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import Field, BaseModel

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash"
)

class Person(BaseModel):
    name: str = Field(description="Name of the person.")
    age: int = Field(description = "Age of the person.")
    occupation: str = Field(description= "Occupation of the person.")

parser = PydanticOutputParser(pydantic_object=Person)

prompt = PromptTemplate(
    template = """
Give me information about a {person}.

{format_instruction}
""",
    input_variables=["person"],
    partial_variables={
        "format_instruction": parser.get_format_instructions()
    }

)

person = input("Enter the name:")

chain = prompt | model | parser
result = chain.invoke({
    "person": person
}
)

print(result)
print(type(result))

