
import os
from functools import lru_cache

from ollama import chat
from langfuse.openai import OpenAI


@lru_cache(maxsize=1)
def get_groq_client():
    return OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ["GROQ_API_KEY"],
    )


def generate_answer(messages: list[dict], temperature: float = 0) -> str:
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()

    if provider == "groq":
        response = get_groq_client().chat.completions.create(
            model=os.getenv("LLM_MODEL", "openai/gpt-oss-20b"),
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""

    if provider == "ollama":
        response = chat(
            model=os.getenv("LLM_MODEL", "llama3.1"),
            messages=messages,
            options={"temperature": temperature},
        )
        return response.message.content or ""

    raise ValueError(f"Неизвестный LLM_PROVIDER: {provider}")
