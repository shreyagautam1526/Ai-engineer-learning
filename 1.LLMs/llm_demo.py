from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# .env file se GEMINI_API_KEY load karega
load_dotenv()

# Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

# LLM ko prompt bhejo
result = llm.invoke(
    "Write a short poem about the beauty of nature."
)

# Response print karo
print(result.content)