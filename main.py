import os
from dotenv import load_dotenv
from agno.agent import Agent
from mistral_model import MistralAgnoModel
from pypdf import PdfReader

load_dotenv()
MISTRAL_KEY = os.getenv("MISTRAL_API_KEY")

if not MISTRAL_KEY:
    raise ValueError(" Pas de clé existante")

mistral_model = MistralAgnoModel(api_key=MISTRAL_KEY, model_id="mistral-small")

pdf = PdfReader("PDF/pwc-ai-analysis.pdf")
full_text = "".join(page.extract_text() or "" for page in pdf.pages)

def chunk_text(text, chunk_size=800, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

chunks = chunk_text(full_text)

def rag_retriever(agent, query, num_documents=3, **kwargs):
    query_lower = query.lower()
    scored_chunks = []
    for i, chunk in enumerate(chunks):
        score = sum(query_lower.count(word) for word in chunk.lower().split())
        scored_chunks.append((score, chunk, i))

    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    top_chunks = scored_chunks[:num_documents]

    print("\n📑 Chunks sélectionnés :")
    for _, chunk, i in top_chunks:
        print(f"\n--- Chunk {i+1} ---\n{chunk[:300]}...\n")

    return [
        {
            "content": chunk,
            "meta_data": {"source": f"pwc-ai-analysis.pdf - chunk {i+1}"}
        }
        for _, chunk, i in top_chunks
    ]

agent = Agent(
    model=mistral_model,
    retriever=rag_retriever,
    search_knowledge=True,
    system_message="Tu es un agent qui répond uniquement à partir du contenu fourni. "
                   "Si la réponse n'est pas dans le document, dis 'Pas trouvé dans le rapport'."
)

prompt = input("Enter Your Prompt: ")
agent.print_response(prompt)
