"""
title: Llama Index Ollama Pipeline
author: open-webui
date: 2024-05-30
version: 1.0
license: MIT
description: A pipeline for retrieving relevant information from a knowledge base using the Llama Index library with Ollama embeddings.
requirements: llama-index, llama-index-llms-ollama, llama-index-embeddings-ollama
"""
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings, VectorStoreIndex, PromptTemplate, Document
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.retrievers import BaseRetriever
from pydantic import BaseModel
from typing import List, Union, Generator, Iterator
import aiohttp
import json
import os



JWT_TOKEN = ""
SERVER_URL = "http://openmetadata.openmetadata.svc.cluster.local:8585/api/v1/tables?limit=10"
async def get_metadata_from_openmetadata_server():
    try:
        headers = {
            "Authorization": f"Bearer {JWT_TOKEN}",
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(
                SERVER_URL,
                headers=headers,
                timeout=10
            ) as response:
                response.raise_for_status()
                return await response.json()
    
    except aiohttp.ClientError as e:
        return f"ClientConnectionError from OpenMetadata server: {e}"
    except aiohttp.ClientPayloadError as e:
        return f"ClientPayloadError from OpenMetadata server: {e}"
    except aiohttp.ClientResponseError as e:
        return f"ClientResponseError from OpenMetadata server: {e}"
    except Exception as e:
        return f"Unexpected error from OpenMetadata server: {e}"


class RAGStringQueryEngine(CustomQueryEngine):
    """RAG String Query Engine."""

    retriever: BaseRetriever
    llm: Ollama
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
                "LLAMAINDEX_MODEL_NAME": os.getenv("LLAMAINDEX_MODEL_NAME", "llama3.2:1"),
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
        #reader = JSONReader(levels_back=0, clean_json=True)
        metadata_json = await get_metadata_from_openmetadata_server()
        metadata_str = json.dumps(metadata_json)
        # This function is called when the server is started.
        global documents, index

        #self.documents = reader.load_data(input_file=metadata_json, extra_info={})
        self.documents = [Document(txt=metadata_str)]
        self.index = VectorStoreIndex.from_documents(self.documents)
        self.retriever = self.index.as_retriever()

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        
        qa_prompt = PromptTemplate(
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "You are a helpful AI Assistant providing company specific knowledge to a new employee."
            "This knowledge is from metadata from the OpenMetadata server."
            "Respond accurately, relying on the available metadata information."
            "If the user asks about a specific table, database, or date, ensure you specify its name and reference relevant metadata when applicable."
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
            return f"ClientPayloadError from ollama server: {e}"
        
        except aiohttp.ClientConnectionError as e:
            return f"ClientConnectionError from ollama server: {e}"

        except aiohttp.ClientResponseError as e:
            return f"ClientResponseError from ollama server: {e}"
        
        except Exception as e:
            return f"Unexpected error from ollama server: {e}"
