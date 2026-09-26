from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash"
)

output_parser = JsonOutputParser()

prompt_template = PromptTemplate(
    template="""
    Give me a review of {movie} with rating and summary.

    {format_instructions}
    """,
    input_variables=["movie"],
    partial_variables={
        "format_instructions": output_parser.get_format_instructions()
    }
)

chain = prompt_template | model | output_parser

# User input
movie = input("Enter movie name: ")

result = chain.invoke({
    "movie": movie
})

print(result)
print(type(result))
print(result["summary"])