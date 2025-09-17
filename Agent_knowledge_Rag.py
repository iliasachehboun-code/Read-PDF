from agno.agent import Agent
from agno.models.xai import xAI
from agno.knowledge import Knowledge
from notion_client import Client
import os

# Notion client
notion = Client(auth=os.getenv("NOTION_TOKEN"))

# Initialize Knowledge object
knowledge_base = Knowledge()

def return_knowledge_base(agent, query, num_documents=None, **kwargs):
    # same logic to query Notion...
    # instead of returning a list, you can add content directly to Knowledge
    matching_docs = []
    # ... your existing matching logic ...
    for doc in matching_documents:
        knowledge_base.add_content(doc["content"], meta_data=doc["meta_data"])
        matching_docs.append(doc)
    return matching_docs
