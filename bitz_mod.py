# pipeline.py (modified OpenWebUI script)
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
                "CHATBOT_API",
                # match service port 80 and service name
                "http://chatbot.openwebui.svc.cluster.local:8181/v1/chat/completions"
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

        # Construct payload with either `message` or OpenAI-style `messages`
        if messages:
            payload = {"model": model_id} if model_id else {}
            payload["messages"] = messages
        else:
            payload = {"message": user_message}
            if model_id:
                payload["model"] = model_id

        try:
            response = requests.post(
                self.valves.CHATBOT_API,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()

            if "response" in result:
                return result["response"]
            elif "choices" in result:
                return result.get("choices", [{}])[0].get("message", {}).get("content", "")
            else:
                return str(result)

        except requests.exceptions.RequestException as e:
            logging.error(f"API request failed: {e}")
            return "Error: Unable to process the request."
