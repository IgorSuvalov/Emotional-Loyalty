import os
from langchain_google_genai import ChatGoogleGenerativeAI


def get_gemini_key():
    try:
        import streamlit as st
        if 'GEMINI_API_KEY' in st.secrets:
            return st.secrets['GEMINI_API_KEY']
    except Exception:
        pass
    key = os.getenv('GEMINI_API_KEY')
    if not key:
        raise RuntimeError("GEMINI_API_KEY not found in environment variables")
    return key


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=get_gemini_key(),
)

