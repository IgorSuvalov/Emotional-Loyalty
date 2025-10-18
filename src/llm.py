import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from langchain_google_genai import ChatGoogleGenerativeAI


def get_gemini_key():
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
                v = st.secrets.get(k)
                if v:
                    return v
    except Exception:
        pass
    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        v = os.getenv(k)
        if v:
            return v
    raise RuntimeError("API key not found. Set 'GEMINI_API_KEY' or 'GOOGLE_API_KEY'.")


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=get_gemini_key(),
)

