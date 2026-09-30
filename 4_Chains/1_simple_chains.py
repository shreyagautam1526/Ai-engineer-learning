from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

prompt = PromptTemplate(
    template = """
Generate a five points about the {topic} in a {style} way.
""",
    input_variables=["topic","style"]
)

parser = StrOutputParser()

model = ChatGoogleGenerativeAI(
    model = "gemini-3.6-flash"
)

topic = input("Enter the name of topic:")
style = input("Enter the style in which you want:")

chain = prompt | model | parser
result = chain.invoke({
    "topic": topic,
    "style": style
})
print(result)
chain.get_graph().print_ascii()