from groq import Groq
from config import GROQ_API_KEY

def generate_answer(question, context, concise=False):
    try:
        client = Groq(api_key=GROQ_API_KEY)

        style = (
            "Give a very short and concise answer (2–3 lines max)."
            if concise else
            "Give a clear and complete answer."
        )

        prompt = f"""
{style}

Answer ONLY using the context below.
If the answer is not present, say: This info isn't in the document.

Context:
{context}

Question:
{question}
"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        msg = str(e).lower()
        if "rate limit" in msg or "429" in msg:
            return "The system is busy right now. Please try again in a few seconds."
        return "An internal error occurred while generating the answer."
