from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableParallel

load_dotenv()

# --------------------------------
# 1. Model
# --------------------------------

model = ChatGoogleGenerativeAI(
    model = "gemini-3.5-flash-lite"
)

parser = StrOutputParser()

# --------------------------------
# 2. Chain1 
# --------------------------------

prompt1 = PromptTemplate(
    template = """
Explain {topic} in simple language.
Keep the explanation beginner-friendly.
""",
    input_variables=["topic"]
)

chain1 = prompt1 | model | parser

# --------------------------------
# 3. Chain2 
# --------------------------------
prompt2 = PromptTemplate(
    template = """
Explain the main advantages of {topic}.
Give 4-5 important points.
""",
    input_variables=["topic"]
)

chain2 = prompt2 | model | parser

# --------------------------------
# 4. Chain3 
# --------------------------------

prompt3 = PromptTemplate(
    template = """
Explain the real-world applications of the {topic}.
Give 4-5 important points.
""",
    input_variables=["topic"]
)

chain3 = prompt3 | model | parser

parallel_chain = RunnableParallel({
    "explanation": chain1,
    "advantages": chain2,
    "applications": chain3

})

topic = input("Enter the topic:")

result = parallel_chain.invoke({
    "topic": topic
})

print("\n" + "=" * 60)
print("EXPLANATION")
print("=" * 60)
print(result["explanation"])


print("\n" + "=" * 60)
print("ADVANTAGES")
print("=" * 60)
print(result["advantages"])


print("\n" + "=" * 60)
print("REAL-WORLD APPLICATIONS")
print("=" * 60)
print(result["applications"])


print("\n" + "=" * 60)
