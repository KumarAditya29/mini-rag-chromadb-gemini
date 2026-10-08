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


def generate_answer(prompt):
    for attempt in range(3):
        try:
            return gemini.models.generate_content(
                model="gemini-3.8-flash",
                contents=prompt
            )
        except Exception as e:
            if attempt == 2:
                raise

            print("Gemini temporarily unavailable. Retrying...")
            time.sleep(3)


def answer_question(question):
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

    response = generate_answer(prompt)

    print("\nANSWER")
    print("------")
    print(response.text)

    print("\nSOURCE CHUNKS")
    print("-------------")

    for i, m in enumerate(metadata, 1):
        print(
            f"{i}. {m['source']} — "
            f"page {m['page']}, chunk {m['chunk']}"
        )


print("Mini RAG ready. Type 'exit' to quit.")

while True:
    question = input("\nQuestion: ").strip()

    if question.lower() == "exit":
        break

    if question:
        answer_question(question)
