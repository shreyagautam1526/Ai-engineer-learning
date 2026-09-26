import warnings

warnings.filterwarnings(
    "ignore",
    message="Direct use of automatic function calling"
)

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash"
)

chat_history = [
    SystemMessage(content="You are a helpful assistant.")
]

while True:

    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    chat_history.append(
        HumanMessage(content=user_input)
    )

    response = model.invoke(chat_history)

    chat_history.append(response)

    content = response.content

    if isinstance(content, list):
        text = "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict)
        )
    else:
        text = str(content)

    print("AI:", text)