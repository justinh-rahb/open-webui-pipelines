"""
title: Google GenAI Manifold Pipeline
author: Marc Lopez (refactor by justinh-rahb)
date: 2024-06-06
version: 1.2
license: MIT
description: A pipeline for generating text using Google's GenAI models in Open-WebUI.
requirements: google-generativeai
environment_variables: GOOGLE_API_KEY
"""

from typing import List, Union, Iterator
import os

from pydantic import BaseModel

import google.generativeai as genai
from google.generativeai.types import GenerationConfig


class Pipeline:
    """Google GenAI pipeline"""

    class Valves(BaseModel):
        """Options to change from the WebUI"""

        GOOGLE_API_KEY: str = ""

    def __init__(self):
        self.type = "manifold"
        self.id = "google_genai"
        self.name = "Google: "

        self.valves = self.Valves(
            **{"GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY", "your-api-key-here")}
        )
        self.pipelines = []
        self.update_api_key()

    def update_api_key(self):
        if self.valves.GOOGLE_API_KEY and self.valves.GOOGLE_API_KEY != "your-api-key-here":
            genai.configure(api_key=self.valves.GOOGLE_API_KEY)
        else:
            print("Warning: GOOGLE_API_KEY is not set or is set to the default value.")

    async def on_startup(self):
        print(f"on_startup:{__name__}")
        self.update_api_key()
        self.update_pipelines()

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")

    async def on_valves_updated(self):
        print(f"on_valves_updated:{__name__}")
        self.update_api_key()
        self.update_pipelines()

    def update_pipelines(self):
        if not self.valves.GOOGLE_API_KEY or self.valves.GOOGLE_API_KEY == "your-api-key-here":
            self.pipelines = [
                {
                    "id": "error",
                    "name": "Google API Key is not set. Please update the API Key in the valves.",
                }
            ]
        else:
            try:
                models = genai.list_models()
                self.pipelines = [
                    {
                        "id": model.name[7:],  # Remove the "models/" prefix
                        "name": model.display_name,
                    }
                    for model in models
                    if "generateContent" in model.supported_generation_methods
                    if model.name.startswith("models/")
                ]
            except Exception as e:
                self.pipelines = [
                    {
                        "id": "error",
                        "name": f"Could not fetch models from Google. Error: {str(e)}",
                    }
                ]
                print(f"Error fetching models: {str(e)}")

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Iterator]:
        if not self.valves.GOOGLE_API_KEY or self.valves.GOOGLE_API_KEY == "your-api-key-here":
            return "Error: GOOGLE_API_KEY is not set or is invalid"

        try:
            # Strip any prefix from the model_id
            model_id = model_id.split('.')[-1]

            # Ensure the model_id starts with "models/"
            if not model_id.startswith("models/"):
                model_id = f"models/{model_id}"

            contents = self.process_messages(messages)

            model = genai.GenerativeModel(model_name=model_id)

            generation_config = GenerationConfig(
                temperature=body.get("temperature", 0.7),
                top_p=body.get("top_p", 0.9),
                max_output_tokens=body.get("max_tokens", 8192),
            )

            response = model.generate_content(
                contents,
                generation_config=generation_config,
                stream=body.get("stream", False),
            )

            if body.get("stream", False):
                return self.stream_response(response)
            else:
                return response.text

        except Exception as e:
            return f"Error: {str(e)}"

    def process_messages(self, messages):
        contents = []
        for message in messages:
            if message["role"] != "system":
                if isinstance(message.get("content"), list):
                    parts = []
                    for content in message["content"]:
                        if content["type"] == "text":
                            parts.append({"text": content["text"]})
                        elif content["type"] == "image_url":
                            image_url = content["image_url"]["url"]
                            if image_url.startswith("data:image"):
                                image_data = image_url.split(",")[1]
                                parts.append({"inline_data": {"mime_type": "image/jpeg", "data": image_data}})
                            else:
                                parts.append({"image_url": image_url})
                    contents.append({"role": message["role"], "parts": parts})
                else:
                    contents.append({
                        "role": "user" if message["role"] == "user" else "model",
                        "parts": [{"text": message["content"]}]
                    })
        return contents

    def stream_response(self, response):
        for chunk in response:
            if chunk.text:
                yield chunk.text
