"""
title: Llama Index Ollama Pipeline
author: open-webui
date: 2024-05-30
version: 1.0
license: MIT
description: A pipeline for retrieving relevant information from a knowledge base using the Llama Index library with Ollama embeddings.
requirements: llama-index, llama-index-llms-ollama, llama-index-embeddings-ollama
"""
from pydantic import BaseModel
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai_like import OpenAILike
from llama_index.core import Settings, VectorStoreIndex, PromptTemplate
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.retrievers import BaseRetriever
from schemas import OpenAIChatMessage
from server.om_server import get_metadata_tables
from typing import List, Union, Generator, Iterator
import aiohttp
import os

from fastapi import FastAPI, HTTPException
#from llama_index.core import Document
from langchain.schema.document import Document
from pydantic import BaseModel
import json
import loggin
import requests
import sys


JWT_TOKEN = "eyJraWQiOiJHYjM4OWEtOWY3Ni1nZGpzLWE5MmotMDI0MmJrOTQzNTYiLCJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJvcGVuLW1ldGFkYXRhLm9yZyIsInN1YiI6ImluZ2VzdGlvbi1ib3QiLCJyb2xlcyI6WyJJbmdlc3Rpb25Cb3RSb2xlIl0sImVtYWlsIjoiaW5nZXN0aW9uLWJvdEBvcGVuLW1ldGFkYXRhLm9yZyIsImlzQm90Ijp0cnVlLCJ0b2tlblR5cGUiOiJCT1QiLCJpYXQiOjE3NDE0ODcwODUsImV4cCI6bnVsbH0.QeHxcKW7LQGaVVnGr9XyZJbvw3-k3ZF1_cHLqIWKyazaPNwsCs0feAphBSwDCs_MvwnWx_k0_hpOth8BBVPccvzv1ZMS-SxYoSF6TvONuchI0s9Z_8l7o4UJcJUQzi_AOjFssvxTwN1NDhZE7N6UX7yDM_9KcKZkS5O9MGNWOkfnqHtA7lM6L6-Yu5oVo9MGMU9PDycm8hQteIdStJXoPM7fXzK6IxghrFuqJ7upXLLqhuFWUyvm6y9H3mUX2Qhnjdbkj6bUx1hpugdWRXu_XWuJPS0WVehGBib2oiUgCFRet5GaLdzYhPke1U-TGBiDXnC5F7rUmplI-saR3hfoXA"


def get_metadata_tables():
    try:
        # Ensure OpenMetadata server is initialized
        #if not initialize_openmetadata_server():
        #    raise HTTPException(status_code=500, detail="Failed to initialize OpenMetadata server")
        # Ensure JWT_TOKEN is available
        if not JWT_TOKEN:
            raise HTTPException(status_code=500, detail="JWT_TOKEN is not set")

        # Set up the headers with the JWT token
        headers = {
            "Authorization": f"Bearer {JWT_TOKEN}",
        }
        # Make a GET request to the OpenMetadata API
        response = requests.get(
            "http://localhost:8585/api/v1/tables?limit=10",
            headers=headers,
            timeout=10  # 10 seconds timeout
        )
        response.raise_for_status()  # Raise an exception for HTTP errors
        metadata_json  = response.json()
        #print(f"Metadata for tables json: {metadata_json}")
        # Convert the JSON to a formatted string
        metadata_str = json.dumps(metadata_json, indent=2)
        #print(f"Metadata for tables json dumps: {metadata_str}")
        table_document = [Document(page_content=metadata_str)]
        #print(f"table document: {table_document}")
        return table_document
    
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error fetching metadata: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    

class RAGStringQueryEngine(CustomQueryEngine):
    """RAG String Query Engine."""

    retriever: BaseRetriever
    llm: OpenAILike
    qa_prompt: PromptTemplate

    def custom_query(self, query_str: str):
        nodes = self.retriever.retrieve(query_str)

        context_str = "\n\n".join([n.node.get_content() for n in nodes])
        response = self.llm.complete(
            self.qa_prompt.format(context_str=context_str, query_str=query_str)
        )

        return str(response)

class Pipeline:

    class Valves(BaseModel):
        LLAMAINDEX_OLLAMA_BASE_URL: str
        LLAMAINDEX_MODEL_NAME: str
        LLAMAINDEX_EMBEDDING_MODEL_NAME: str

    def __init__(self):
        self.name = "Ollama-ChatBot-Pipeline"
        self.documents = None
        self.index = None
        self.retriever = None

        self.valves = self.Valves(
            **{
                "LLAMAINDEX_OLLAMA_BASE_URL": os.getenv("LLAMAINDEX_OLLAMA_BASE_URL", "http://localhost:11434"),
                "LLAMAINDEX_MODEL_NAME": os.getenv("LLAMAINDEX_MODEL_NAME", "llama3.2:1b"),
                "LLAMAINDEX_EMBEDDING_MODEL_NAME": os.getenv("LLAMAINDEX_EMBEDDING_MODEL_NAME", "nomic-embed-text"),
            }
        )

    async def on_startup(self):

        Settings.embed_model = OllamaEmbedding(
            model_name=self.valves.LLAMAINDEX_EMBEDDING_MODEL_NAME,
            base_url=self.valves.LLAMAINDEX_OLLAMA_BASE_URL,
        )
        Settings.llm = Ollama(
            model=self.valves.LLAMAINDEX_MODEL_NAME,
            base_url=self.valves.LLAMAINDEX_OLLAMA_BASE_URL,
        )

        # This function is called when the server is started.
        global documents, index

        self.documents = get_metadata_tables()
        self.index = VectorStoreIndex.from_documents(self.documents)
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom RAG pipeline.
        # Typically, you would retrieve relevant information from your knowledge base and synthesize it to generate a response.

        qa_prompt = PromptTemplate(
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "You are a helpful AI Assistant providing company specific knowledge to a new employee."
            "This knowledge is from metadata from the OpenMetadata server."
            Respond accurately, relying on the available metadata information.
            "If the user asks about a specific table, database, or date, ensure you specify its name and reference relevant metadata when applicable"
            "Generate human readable output, avoid creating output with gibberish text."
            "Generate only the requested output, don't include any other language before or after the requested output."
            "Provide complete and detailed answers whenever possible"
            "Use the same language as user does."
            "Never say thank you, that you are happy to help, that you are an AI agent, etc. Just answer directly."
            "Never generate offensive or foul language."
            "Given the context information and not prior knowledge, "
            "answer the query.\n"
            "Query: {query_str}\n"
            "Answer: "
        )

        self.retriever = self.index.as_retriever()
        query_engine = RAGStringQueryEngine(
            retriever=self.retriever,
            llm=Settings.llm,
            qa_prompt=qa_prompt,
        )


        #query_engine = self.index.as_query_engine(streaming=True)
        #response = query_engine.query(user_message)
        #return response.response_gen

        try:
            response = query_engine.custom_query(user_message)

            if hasattr(response, 'response_gen'):
                # final_response = self.handle_streaming_response(response.response_gen)
                return str(response)
            else:
                # final_response = response.response
                return str(response)
            
        except aiohttp.ClientPayloadError as e:
            return f"ClientPayloadError: {e}"
        
        except aiohttp.ClientConnectionError as e:
            return f"ClientConnectionError: {e}"

        except aiohttp.ClientResponseError as e:
            return f"ClientResponseError: {e}"
        
        except Exception as e:
            return f"Unexpected error: {e}"
