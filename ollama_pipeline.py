"""
title: Llama Index ↔ OpenMetadata RAG Pipeline
author: open-webui (refactor)
date: 2025-05-15
version: 1.4
license: MIT
description: A pipeline for RAG over OpenMetadata using Llama Index + Ollama.
requirements: llama-index-core, llama-index-llms-ollama, llama-index-embeddings-ollama, pydantic>=2.0, aiohttp
"""

import os
import json
from typing import Dict, List, Union, Generator, Iterator

import aiohttp
from pydantic import BaseModel

from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import Document, VectorStoreIndex, PromptTemplate
from llama_index.core import Settings
from llama_index.core.node_parser import JSONNodeParser
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import CondensePlusContextChatEngine

# -----------------------------------------------------------------------------
# Async helper to fetch raw JSON metadata from OpenMetadata
# -----------------------------------------------------------------------------
async def fetch_all_tables(endpoint: str, headers: Dict[str, str], limit: int = 100) -> List[Dict]:
    tables: List[Dict] = []
    after = None
    async with aiohttp.ClientSession() as session:
        while True:
            params = {"limit": limit, **({"after": after} if after else {})}
            async with session.get(endpoint, headers=headers, params=params) as resp:
                resp.raise_for_status()
                payload = await resp.json()

            tables.extend(payload.get("data", []))
            after = payload.get("paging", {}).get("after")
            if not after:
                break

    return tables

# -----------------------------------------------------------------------------
# Parse raw JSON into TextNode chunks
# -----------------------------------------------------------------------------
def parse_tables_to_nodes(raw_tables: List[Dict]) -> List:
    """
    Given a list of raw table JSON dicts, return a flat list of parsed nodes (dict or TextNode) via JSONNodeParser.
    """
    parser = JSONNodeParser()
    all_nodes = []
    for tbl in raw_tables:
        # Create a Document wrapping raw JSON
        json_doc = Document(id_=tbl.get("fullyQualifiedName"), text=json.dumps(tbl))
        # Parse into nodes (these may be dicts or TextNode instances)
        nodes = parser.get_nodes_from_documents([json_doc])
        all_nodes.extend(nodes)
    return all_nodes

class Pipeline:

    class Valves(BaseModel):
        # Ollama
        LLAMAINDEX_OLLAMA_BASE_URL: str
        LLAMAINDEX_MODEL_NAME: str
        LLAMAINDEX_EMBEDDING_MODEL_NAME: str

        # OpenMetadata
        OM_API: str
        OM_API_ENDPOINT: str
        OM_JWT_TOKEN: str

    def __init__(self):
        # load everything from environment/.env
        env = {
            "LLAMAINDEX_OLLAMA_BASE_URL": os.getenv(
                "LLAMAINDEX_OLLAMA_BASE_URL", "http://ollama-dev.ollama.svc.cluster.local:11434"
            ),
            "LLAMAINDEX_MODEL_NAME": os.getenv("LLAMAINDEX_MODEL_NAME", "llama3.1:8b"),
            "LLAMAINDEX_EMBEDDING_MODEL_NAME": os.getenv(
                "LLAMAINDEX_EMBEDDING_MODEL_NAME", "bge-m3:567m"
            ),
            "OM_API": os.getenv(
                "OM_API", "http://openmetadata.openmetadata.svc.cluster.local:8585/api/v1/"
            ),
            "OM_API_ENDPOINT": os.getenv("OM_ENDPOINT", "tables"),
            "OM_JWT_TOKEN": os.getenv("OM_JWT_TOKEN", ""),
        }
        self.valves = self.Valves(**env)
        self.chat_engine = None
        self.documents: List[str] = []
        self.index: VectorStoreIndex = None  # type: ignore

    async def on_startup(self):
        """Configure Ollama, connect to OpenMetadata, fetch samples, build index."""
        # 1) wire up Ollama in llama-index
        Settings.embed_model = OllamaEmbedding(
            model_name=self.valves.LLAMAINDEX_EMBEDDING_MODEL_NAME,
            base_url=self.valves.LLAMAINDEX_OLLAMA_BASE_URL,
        )
        Settings.llm = Ollama(
            model=self.valves.LLAMAINDEX_MODEL_NAME,
            base_url=self.valves.LLAMAINDEX_OLLAMA_BASE_URL,
        )

        # 2) Fetch raw metadata
        table_endpoint = self.valves.OM_API + self.valves.OM_API_ENDPOINT
        headers = {"Authorization": f"Bearer {self.valves.OM_JWT_TOKEN}","Content-Type": "application/json"} 
        raw_tables = await fetch_all_tables(table_endpoint, headers, limit=100)

        # 3) Parse into TextNode chunks
        nodes = parse_tables_to_text_nodes(raw_tables)

        # 4) Build index directly from TextNodes
        self.index = VectorStoreIndex(nodes)

        # 5) Initialize chat engine with memory
        memory = ChatMemoryBuffer.from_defaults(token_limit=3900)
        context_prompt = (
            "You are a chatbot conversant with the organization's metadata."
            " Use the context below to answer user queries. If you don’t know, say you don’t know."  
            "\nContext:\n{context_str}\nInstruction: {query_str}\n"
        )
        self.chat_engine = CondensePlusContextChatEngine.from_defaults(
            self.index.as_retriever(),
            memory=memory,
            llm=Settings.llm,
            context_prompt=context_prompt,
            verbose=True,
        )
    
    async def on_shutdown(self):
        """No‐op cleanup."""
        pass

    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: List[dict],
        body: dict,
    ) -> Union[str, Generator, Iterator]:
        """Stream the Llama Index response for a user query with OpenMetadata context."""
        # Print logs
        print("chat history:", messages)
        print("incoming:", user_message)
        return self.chat_engine.chat(user_message)
