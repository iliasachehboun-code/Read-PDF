import os
from dotenv import load_dotenv
from agno.agent import Agent
from agno.knowledge import Knowledge
from mistralai import Mistral
from notion_client import Client

# Charger les variables d'environnement
load_dotenv()

# Récupérer les clés API
MISTRAL_KEY = os.getenv("MISTRAL_API_KEY")
NOTION_KEY = os.getenv("NOTION_TOKEN")

# Clients
client = Mistral(api_key=MISTRAL_KEY)
notion = Client(auth=NOTION_KEY)

# Base de connaissances
knowledge_base = Knowledge()

# --- Fonction wrapper pour appeler Mistral ---
def mistral_responder(prompt: str):
    response = client.chat.complete(
        model="mistral-small",  # ou mistral-medium / mistral-large
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Création de l’agent avec Mistral comme modèle
agent = Agent(
    model=mistral_responder,
    knowledge=knowledge_base,
    search_knowledge=True,
    instructions="Tu es un assistant qui utilise la base de connaissances Notion quand c’est pertinent."
)

# --- Récupération depuis Notion ---
def return_knowledge_base(agent, query, num_documents=5, **kwargs):
    """
    Cherche dans Notion, ajoute les résultats dans la base de connaissances,
    et retourne les documents correspondants.
    """
    search = notion.search(query=query, page_size=num_documents)
    matching_docs = []
    for r in search.get("results", []):
        title = r.get("properties", {}).get("title", {}).get("title", [{}])[0].get("plain_text", "Untitled")
        page_id = r.get("id")
        content = f"{title} (id: {page_id})"
        doc = {
            "content": content,
            "meta_data": {"source": "notion", "id": page_id, "title": title}
        }
        knowledge_base.add_content(doc["content"], meta_data=doc["meta_data"])
        matching_docs.append(doc)
    return matching_docs
