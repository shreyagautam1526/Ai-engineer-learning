from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0
)

st.header("Research Tool")

user_input = st.text_input("Enter your research query:")

if st.button("Summarize"):
    result = model.invoke(user_input)

    content = result.content

    if isinstance(content, list):
        text = "\n".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    else:
        text = str(content)

    st.write(text)