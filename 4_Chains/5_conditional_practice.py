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

payment_prompt = PromptTemplate(
    template = """
    You are a customer support assistant.
    Customer problem:{message}
    Give a helpful response for this payment-related issue.
    """,
    input_variables=["message"]
)
payment_chain = payment_prompt | model | parser

order_prompt = PromptTemplate(
    template = """
    You are a customer support assistant.
    Customer problem:{message}
    Give helpful response for this order-related issue.
    """,
    input_variables=["message"]
)
order_chain = order_prompt | model | parser

refund_prompt = PromptTemplate(
    template = """
    You are a customer support assistant.
    Customer problem:{message}
    Give helpful response for this refund-related issue.
    """,
    input_variables=["message"]
)
refund_chain = refund_prompt | model | parser

default_prompt = PromptTemplate(
    template="""
    You are a customer support assistant.

    The customer entered an unsupported category.
    Politely tell them to choose one of:
    payment, order, refund.
    """,
    input_variables=[]
)

default_chain = default_prompt | model | parser

branch = RunnableBranch(
    (
        lambda x: x["category"].lower() == "payment",
        payment_chain
    ),
    (
        lambda x: x["category"].lower() == "order",
        order_chain
    ),
    (
        lambda x: x["category"].lower() == "refund",
        refund_chain
    ),
    default_chain
)

category = input("Enter the category:")
message = input("Enter the message:")

result = branch.invoke(
    {
        "category": category,
        "message": message
    }
)

print(result)

