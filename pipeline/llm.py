import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

_client = OpenAI(
    api_key=os.environ["LLM_API_KEY"],
    base_url=os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
)
_MODEL = os.environ.get("MODEL", "llama3-8b-8192")

_SYSTEM_PROMPT = (
    "You are a compassionate care advisor helping elderly and vulnerable individuals "
    "access social services and support. Respond with empathy, clarity, and practical "
    "guidance. Keep responses concise (2-4 sentences). Do not mention any organisation "
    "or product names. If the input contains placeholder tokens like [PERSON_1] or "
    "[NRIC_1], treat them as real values in your response."
)


def call_llm(text: str) -> str:
    response = _client.chat.completions.create(
        model=_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        temperature=0.4,
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()
