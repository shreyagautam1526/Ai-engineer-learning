from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.6-flash")
output_parser = StrOutputParser()

prompt_template = PromptTemplate(
    template = "Tell me about {topic} in a detailed manner with specific points and examples.",
    input_variables = ["topic"]
)
prompt_template1 = PromptTemplate(
    template = "Provide a comprehensive summary of following text .\n {text} including key aspects and relevant details.",
    input_variables = ["text"]
)

chain = prompt_template | model | output_parser | prompt_template1 | model | output_parser

result = chain.invoke({"topic": "the impact of climate change on global ecosystems"})

print(result)
