import os
import time
import chromadb
from dotenv import load_dotenv
from google import genai

load_dotenv()

chroma = chromadb.PersistentClient(path="chroma_db")
collection = chroma.get_collection("documents")

gemini = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
)

# Generation model. Override with the RAG_MODEL environment variable.
RAG_MODEL = os.getenv("RAG_MODEL", "gemini-3.8-flash")


def generate_answer(prompt, model=None):
    for attempt in range(3):
        try:
            return gemini.models.generate_content(
                model=model or RAG_MODEL,
                contents=prompt
            )
        except Exception as e:
            if attempt == 2:
                raise

            print("Gemini temporarily unavailable. Retrying...")
            time.sleep(3)


def run_rag(question, model=None):
    """Run the full RAG pipeline and return structured results.

    model overrides the generation model (default: RAG_MODEL).

    Returns a dict with:
      answer  - Gemini's generated answer
      model   - the generation model that produced the answer
      context - the exact context string passed to Gemini
      chunks  - the retrieved chunk texts
      sources - metadata (source, page, chunk) for each retrieved chunk
    """
    results = collection.query(
        query_texts=[question],
        n_results=3
    )

    documents = results["documents"][0]
    metadata = results["metadatas"][0]

    context = "\n\n".join(
        f"[Source: {m['source']}, page {m['page']}]\n{doc}"
        for doc, m in zip(documents, metadata)
    )

    prompt = f"""You are a helpful RAG assistant.

Answer the question using ONLY the provided context.

If the context does not contain enough information, say:
"I don't have enough information in the provided documents."

Do not use outside knowledge.

Context:
{context}

Question:
{question}

Give a concise and accurate answer.
"""

    model = model or RAG_MODEL
    response = generate_answer(prompt, model)

    return {
        "answer": response.text,
        "model": model,
        "context": context,
        "chunks": documents,
        "sources": metadata,
    }


def answer_question(question):
    result = run_rag(question)

    print("\nANSWER")
    print("------")
    print(result["answer"])

    print("\nSOURCE CHUNKS")
    print("-------------")

    for i, m in enumerate(result["sources"], 1):
        print(
            f"{i}. {m['source']} — "
            f"page {m['page']}, chunk {m['chunk']}"
        )


def main():
    print("Mini RAG ready. Type 'exit' to quit.")

    while True:
        question = input("\nQuestion: ").strip()

        if question.lower() == "exit":
            break

        if question:
            answer_question(question)


if __name__ == "__main__":
    main()
