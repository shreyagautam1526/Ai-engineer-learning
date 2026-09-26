import warnings

warnings.filterwarnings(
    "ignore",
    message="Direct use of automatic function calling"
)


from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash"
)

chat_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful assistant.

When your answer contains mathematical formulas:
- Use LaTeX notation.
- Use $...$ for inline formulas.
- Use $$...$$ for formulas that should appear on a separate line.
- Keep formulas properly formatted and readable."""
    ),
    MessagesPlaceholder(variable_name = "chat_history"),
    ('human', "Explain this {topic} in {style} way.")
])

chat_history = []

while True:

    topic = input("Enter a topic (or type 'exit' to quit): ")
    if topic.lower() == "exit":
        break

    style = input("Enter a style (e.g., formal, casual, humorous, summary, medium, long, short): ")

    chat_history.append(HumanMessage(content="Explain this {topic} in {style} way."))

    prompt = chat_prompt.format_messages(chat_history=chat_history, topic=topic, style=style)

    response = model.invoke(prompt)

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