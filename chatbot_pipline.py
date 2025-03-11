from typing import List, Optional
from pydantic import BaseModel
import requests
import os
import logging

logging.basicConfig(level=logging.INFO)

class Pipeline:
    class Valves(BaseModel):
        OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    def __init__(self):
        self.name = "Custom Chat Pipeline"
        self.valves = self.Valves()
        self.api_endpoint = "http://localhost:5000/v1/chat/completions"

    async def on_startup(self):
        logging.info(f"Pipeline '{self.name}' started.")

    async def on_shutdown(self):
        logging.info(f"Pipeline '{self.name}' shutting down.")

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Optional[str]:
        logging.info(f"Processing message: {user_message}")

        # Construct payload for the API request
        payload = {
            "messages": [
    {"role": "user", "content": user_message }]
        }

        try:
            response = requests.post(self.api_endpoint, json=payload, headers={"Content-Type": "application/json"})
            response.raise_for_status()  # Raises an error for non-200 responses

            result = response.json()
            return result.get("choices", [{}])[0].get("message", {}).get("content", "No response received.")

        except requests.exceptions.RequestException as e:
            logging.error(f"API request failed: {e}")
            return "Error: Unable to process the request."






#pipeline = Pipeline()
#result = pipeline.pipe(user_message="Hello", model_id="", messages=[{"role": "user", "content": "Hello"}], body={})
#print(result)
