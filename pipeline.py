"""
title: Llama Index ↔ OpenMetadata RAG Pipeline
author: open-webui (refactor)
date: 2025-05-15
version: 1.4
license: MIT
description: A pipeline for RAG over OpenMetadata using Llama Index + Ollama.
requirements: llama-index, llama-index-llms-ollama, llama-index-embeddings-ollama, openmetadata-ingestion==1.7.0.0, pydantic
"""

import os
from typing import List, Union, Generator, Iterator

from pydantic import BaseModel
from schemas import OpenAIChatMessage  # your chat message schema

from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

# OpenMetadata SDK imports
from metadata.generated.schema.entity.services.connections.metadata.openMetadataConnection import (
    OpenMetadataConnection,
    AuthProvider,
)
from metadata.generated.schema.security.client.openMetadataJWTClientConfig import (
    OpenMetadataJWTClientConfig,
)
from metadata.ingestion.ometa.ometa_api import OpenMetadata
from metadata.ingestion.ometa.mixins.table_mixin import OMetaTableMixin

# Entity classes
from metadata.generated.schema.entity.data.database import Database
from metadata.generated.schema.entity.data.table import Table
#from metadata.generated.schema.entity.data.metric import Metric
#from metadata.generated.schema.entity.data.dashboard import Dashboard
#from metadata.generated.schema.entity.data.report import Report
#from metadata.generated.schema.entity.services.ingestionPipelines.ingestionPipeline import (
#    IngestionPipeline,
#)
#from metadata.generated.schema.entity.messaging.topic import Topic


class Pipeline:

    class Valves(BaseModel):
        # Ollama
        LLAMAINDEX_OLLAMA_BASE_URL: str
        LLAMAINDEX_MODEL_NAME: str
        LLAMAINDEX_EMBEDDING_MODEL_NAME: str

        # OpenMetadata
        OM_HOST_PORT: str
        OM_JWT_TOKEN: str

    def __init__(self):
        # load everything from environment/.env
        env = {
            "LLAMAINDEX_OLLAMA_BASE_URL": os.getenv(
                "LLAMAINDEX_OLLAMA_BASE_URL", "http://localhost:11434"
            ),
            "LLAMAINDEX_MODEL_NAME": os.getenv("LLAMAINDEX_MODEL_NAME", "llama3.2"),
            "LLAMAINDEX_EMBEDDING_MODEL_NAME": os.getenv(
                "LLAMAINDEX_EMBEDDING_MODEL_NAME", "nomic-embed-text"
            ),
            "OM_HOST_PORT": os.getenv(
                "OM_HOST_PORT", "http://localhost:8585/api"
            ),
            "OM_JWT_TOKEN": os.getenv("OM_JWT_TOKEN", ""),
        }
        self.valves = self.Valves(**env)

        self.documents: List[str] = []
        self.index: VectorStoreIndex = None  # type: ignore
        self.metadata: OpenMetadata = None   # type: ignore

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

        # 2) connect to OpenMetadata
        conn = OpenMetadataConnection(
            hostPort=self.valves.OM_HOST_PORT,
            authProvider=AuthProvider.openmetadata,
            securityConfig=OpenMetadataJWTClientConfig(
                jwtToken=self.valves.OM_JWT_TOKEN
            ),
        )
        self.metadata = OpenMetadata(conn)
        table_mixin = OMetaTableMixin(self.metadata)

        # 3) collect docs across your seven APIs
        # — Databases
        for db in self.metadata.list_all_entities(Database, limit=50):
            self.documents.append(f"Database: {db.name} — {db.description or ''}")

        # — Tables: up to 20 sample rows each
        for tbl in self.metadata.list_all_entities(Table, limit=50):
            try:
                df = table_mixin.get_table_sample_data(
                    table_fqn=tbl.fullyQualifiedName, limit=20
                )
            except Exception:
                continue
            for row in df.to_dict(orient="records"):
                snippet = " | ".join(f"{k}: {v}" for k, v in row.items())
                self.documents.append(f"Table [{tbl.name}]: {snippet}")
        """
        # — Metrics, Dashboards, Reports, Pipelines, Topics
        for entity_cls, label in [
            (Metric, "Metric"),
            (Dashboard, "Dashboard"),
            (Report, "Report"),
            (IngestionPipeline, "Pipeline"),
            (Topic, "Topic"),
        ]:
            for ent in self.metadata.list_all_entities(entity_cls, limit=50):
                self.documents.append(f"{label}: {ent.name} — {ent.description or ''}")
        """
        # 4) build the vector‐store index
        self.index = VectorStoreIndex.from_documents(self.documents)

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
        """
        Stream the Llama Index response for a user query.
        """
        print("chat history:", messages)
        print("incoming:", user_message)

        query_engine = self.index.as_query_engine(streaming=True)
        response = query_engine.query(user_message)
        return response.response_gen
