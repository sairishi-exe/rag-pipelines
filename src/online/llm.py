import requests

from src.config import OLLAMA_MODEL, OLLAMA_URL


def build_prompt(question: str, chunks: list[dict]) -> str:
    """Format retrieved chunks and question into a prompt for the LLM."""
    context = "\n".join(
        f"[{i}] {chunk['text']}" for i, chunk in enumerate(chunks, 1)
    )

    return (
        "You are a biomedical expert. Answer the following question in a few words "
        "based only on the provided context. If the context does not contain the answer, "
        'say "unanswerable".\n\n'
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def generate_answer(prompt: str) -> str:
    """Call Ollama's REST API and return the generated text."""
    response = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    })
    response.raise_for_status()
    return response.json()["response"].strip()


def answer_question(question: str, chunks: list[dict]) -> str:
    """Build prompt from question + chunks, send to LLM, return answer."""
    prompt = build_prompt(question, chunks)
    return generate_answer(prompt)
