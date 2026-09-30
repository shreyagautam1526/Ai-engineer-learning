from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

model = ChatGoogleGenerativeAI(
    model = "gemini-3.5-flash-lite"
)

prompt1 = PromptTemplate(
    template = """
Explain {topic} in simple language.

Keep the explanation beginner-friendly.
""",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template = """
Summarize the following explanation in 2-3 lines:

{explanation}
""",
    input_variables=["explanation"]
)

parser = StrOutputParser()

topic = input("Enter the topic:")
explanation = input("Enter the type of explanation:")

chain = prompt1 | model | parser | prompt2 | model | parser
result = chain.invoke({
    "topic": topic,
    "explanation": explanation
})

print("\n"+"="*50)
print(result)
print("\n" + "=" * 50)
chain.get_graph().print_ascii()