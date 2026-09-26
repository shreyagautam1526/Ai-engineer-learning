from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import (
    StructuredOutputParser,
    ResponseSchema
)
load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash"
)
response = [
    ResponseSchema(
        name="programming_languages",
        description="A list of programming languages known by candidate."
    ),
    ResponseSchema(
        name="databases",
        description="A list of databases known by candidate."
    ),
    ResponseSchema(
        name="ai tools",
        description="A list of ai tools known by candidate."
    ),
    ResponseSchema(
        name="frameworks",
        description="A list of frameworks known by candidate."
    ),
    ResponseSchema(
        name="projects",
        description="A list of projects made by candidate."
    )
]

# Create the Structure output parser
parser = StructuredOutputParser.from_response_schemas(response)

#create prompt
prompt = PromptTemplate(
    template= """
Extract the candidate's skills and projects from the following resume:

{resume},
    
{format_instructions}""",
input_variables=["resume"],
partial_variables={
    "format_instructions":parser.get_format_instructions()
}
)

resume = input("Enter the resume text: ")

chain = prompt | model | parser
result = chain.invoke({
    "resume": resume
})
print("\nExtracted Information:")
print(result)

print("\nType:")
print(type(result))
