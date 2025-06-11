"""
title: Llama Index ↔ OpenMetadata RAG Pipeline
author: open-webui (refactor)
date: 2025-05-15
version: 1.4
license: MIT
description: A pipeline for RAG over OpenMetadata using Llama Index + Ollama.
requirements: logging, pydantic, typing
"""
from typing import List, Optional
from pydantic import BaseModel
import requests
import os
import logging

logging.basicConfig(level=logging.INFO)

class Pipeline:
    class Valves(BaseModel):
        CHATBOT_API: str 

    def __init__(self):
        env = {
            "CHATBOT_API": os.getenv(
                "CHATBOT_API", "http://chatbot.openwebui.svc.cluster.local:8000/v1/chat/completions"
                ),
        }
        self.name = "visionr"
        self.valves = self.Valves(**env)

    async def on_startup(self):
        logging.info(f"Pipeline '{self.name}' started.")

    async def on_shutdown(self):
        logging.info(f"Pipeline '{self.name}' shutting down.")

    def pipe(
        self,
        user_message: str,
        model_id: str = "",
        messages: Optional[List[dict]] = None,
        body: Optional[dict] = None
    ) -> Optional[str]:
        logging.info(f"Processing message: {user_message}")
    
        # Construct payload matching your working cURL example
        payload = {
            "message": user_message,
            # Include model_id if provided (some local servers require this)
            **({"model": model_id} if model_id else {})
        }
    
        try:
            response = requests.post(
                self.valves.CHATBOT_API,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            # Log the full response for debugging
            logging.debug(f"API Response: {response.text}")
            
            response.raise_for_status()  # Raises an error for non-200 responses
    
            result = response.json()
            
            # Handle different possible response structures
            if "response" in result:
                return result["response"]
            elif "choices" in result:  # OpenAI-compatible format
                return result.get("choices", [{}])[0].get("message", {}).get("content", "")
            else:
                return str(result)  # Fallback to string representation
    
        except requests.exceptions.RequestException as e:
            logging.error(f"API request failed: {e}")
            if hasattr(e, 'response') and e.response:
                logging.error(f"Error details: {e.response.text}")
            return "Error: Unable to process the request."

# Example usage
#if __name__ == "__main__":
    #pipeline = Pipeline()
    #result = pipeline.pipe(
        #user_message="What databases present?",
        #model_id=""  # Optional if your server doesn't require it
    #)
    #print(result)
