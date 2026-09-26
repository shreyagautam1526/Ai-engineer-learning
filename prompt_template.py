from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0
)

paper_input = st.selectbox(
    "Select a Research Paper:",
    [
        "1. The Attention is all you need",
        "2. Selective and cross-reactive SARS-CoV-2 T cell epitopes in unexposed humans",
        "3. Active escape of prey from predator vent via the digestive tract"
    ]
)

style_input = st.selectbox(
    "Select a Style:",
    [
        "1. Summarize",
        "2. Explain",
        "3. Critique"
    ]
)

style_length = st.selectbox(
    "Select a Length:",
    [
        "1. Short",
        "2. Medium",
        "3. Long"
    ]
)

st.header("Research Tool")

# Create the prompt template
prompt_template = PromptTemplate(
    template="""
You are a research assistant.

Your task is to {style_input} the research paper titled
"{paper_input}" in a {style_length} manner.

Please provide a clear and concise response.
""",
input_variables=["style_input", "paper_input", "style_length"],
validate_template=True
)

if st.button("Summarize"):

    # Fill the placeholders
    prompt = prompt_template.invoke({
        "style_input": style_input,
        "paper_input": paper_input,
        "style_length": style_length
    })

    # Send the prompt to Gemini
    result = model.invoke(prompt)

    # Extract model response
    content = result.content

    # Handle structured content
    if isinstance(content, list):
        text = "\n".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    else:
        text = str(content)

    st.write(text)