from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableBranch

load_dotenv()

# --------------------------------
# 1. Model
# --------------------------------

model = ChatGoogleGenerativeAI(
    model = "gemini-3.5-flash-lite"
)

parser = StrOutputParser()

beginner_prompt = PromptTemplate(
    template = "" \
    "Explain the {topic} in very simple language.",
    input_variables = ["topic"]
)
beginner_chain = beginner_prompt | model | parser

intermediate_prompt = PromptTemplate(
    template = "" \
    "Explain the {topic} with technical details.",
    input_variables=["topic"]
)
intermediate_chain = intermediate_prompt | model | parser

advanced_prompt = PromptTemplate(
    template = "" \
    "Explain the {topic} in depth with implementation details.",
    input_variables=["topic"]
)
advanced_chain = advanced_prompt | model | parser

default_prompt = PromptTemplate(
    template="""
Please enter a valid level.

Valid levels are:
- beginner
- intermediate
- advanced
""",
    input_variables=[]
)

default_chain = default_prompt | model | parser

branch = RunnableBranch(
    (
        lambda x: x["level"].lower() == "beginner",
        beginner_chain
    ),
    (
        lambda x: x["level"].lower() == "intermediate",
        intermediate_chain
    ),
    (
        lambda x: x["level"].lower() == "advanced",
        advanced_chain
    ),
    default_chain
)

topic = input("Enter the topic:")
level = input("Enter the level(beginner/intermediate/advanced):")

result = branch.invoke(
    {
        "topic": topic,
        "level": level
    }
)

print(result)
