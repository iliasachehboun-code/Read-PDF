import os
from dotenv import load_dotenv
from agno.agent import Agent
from agno.knowledge import Knowledge
from mistralai import Mistral

load_dotenv()

MISTRAL_KEY = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=MISTRAL_KEY)

knowledge_base = Knowledge()

def mistral_responder(prompt: str):
    response = client.chat.complete(
        model="mistral-small", 
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

agent = Agent(
    model=mistral_responder,
    knowledge=knowledge_base,
    search_knowledge=True,
    instructions="Tu es un assistant qui utilise la base de connaissances Notion quand c’est pertinent."
)

