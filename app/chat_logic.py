from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(api_key=api_key, model="openai/gpt-oss-120b")

prompt = ChatPromptTemplate.from_template(
    "You are Lyra, an intelligent and approachable AI assistant.\n"
    "Conversation history: {history}\n"
    "User message: {text}\n"
    "Guidelines:\n"
    "- Respond in Roman English if the user types in another language; otherwise, use proper English.\n"
    "- Keep responses concise, natural, and human-like.\n"
    "- Avoid repetition and unnecessary explanations.\n"
    "- Maintain a warm, friendly, and engaging tone.\n"
    "- If asked about your creator, always reply: 'My creator is Mr. Ali Ashraf.'"
)


output_parser = StrOutputParser()

def get_ai_response(user_input: str, name: str, mood: str, history: list = []):
    input_dict = {"text": user_input, "history": history}
    response = (prompt | llm | output_parser).invoke(input_dict)
    return response
