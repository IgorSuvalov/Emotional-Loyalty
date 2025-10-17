from langchain_google_genai import ChatGoogleGenerativeAI
from src.apikey import GEMINI_API_KEY

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GEMINI_API_KEY,
)

