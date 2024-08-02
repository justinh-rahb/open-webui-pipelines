"""
title: Google GenAI Manifold Pipeline
author: justinh-rahb
date: 2024-08-02
version: 0.1.0
license: MIT
description: A pipeline for generating text using Google's VertexAI models in Open-WebUI.
requirements: google-generativeai
environment_variables: GOOGLE_PROJECT_ID, SERVICE_ACCOUNT_FILE, USE_PERMISSIVE_SAFETY, DEFAULT_MODEL, REGION, MAX_OUTPUT_TOKENS
"""

import os
import json
import asyncio
import aiohttp
from typing import List, Union, Optional, Iterator, AsyncIterator
from tenacity import retry, stop_after_attempt, wait_exponential

from google.oauth2 import service_account
from google.auth.transport.requests import Request

from pydantic import BaseModel, Field

class Pipeline:
    """Google Vertex AI pipeline for open-webui"""

    class Valves(BaseModel):
        """Options to change from the WebUI"""
        GOOGLE_PROJECT_ID: str = Field(default="", description="Google Cloud Project ID")
        SERVICE_ACCOUNT_FILE: str = Field(default="path/to/service-account.json", description="Path to service account JSON file")
        USE_PERMISSIVE_SAFETY: bool = Field(default=False, description="Use permissive safety settings")
        DEFAULT_MODEL: str = Field(default="", description="Default model to use")
        REGION: str = Field(default="us-central1", description="GCP region for Vertex AI services")
        MAX_OUTPUT_TOKENS: int = Field(default=1024, description="Maximum number of output tokens")

    def __init__(self):
        self.type = "manifold"
        self.id = "vertex_ai"
        self.name = "Google Vertex AI"

        self.valves = self.Valves(**{
            "GOOGLE_PROJECT_ID": os.getenv("GOOGLE_PROJECT_ID", ""),
            "SERVICE_ACCOUNT_FILE": os.getenv("SERVICE_ACCOUNT_FILE", "path/to/service-account.json"),
            "USE_PERMISSIVE_SAFETY": False,
            "DEFAULT_MODEL": os.getenv("DEFAULT_MODEL", ""),
            "REGION": os.getenv("REGION", "us-central1"),
            "MAX_OUTPUT_TOKENS": int(os.getenv("MAX_OUTPUT_TOKENS", 1024))
        })
        self.pipelines = []
        self.credentials = None
        self.access_token = None
        self.session = None
        self.operations = {}  # To track long-running operations

        self.authenticate_service_account()
        asyncio.run(self.update_pipelines())

    def authenticate_service_account(self):
        """Authenticate using service account credentials"""
        try:
            self.credentials = service_account.Credentials.from_service_account_file(
                self.valves.SERVICE_ACCOUNT_FILE,
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            self.credentials.refresh(Request())
            self.access_token = self.credentials.token
        except Exception as e:
            print(f"Authentication error: {e}")
            self.access_token = None

    async def on_startup(self) -> None:
        """This function is called when the server is started."""
        print(f"on_startup:{__name__}")
        self.authenticate_service_account()
        self.session = aiohttp.ClientSession()
        await self.update_pipelines()

    async def on_shutdown(self) -> None:
        """This function is called when the server is stopped."""
        print(f"on_shutdown:{__name__}")
        if self.session:
            await self.session.close()

    async def on_valves_updated(self) -> None:
        """This function is called when the valves are updated."""
        print(f"on_valves_updated:{__name__}")
        self.authenticate_service_account()
        await self.update_pipelines()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def update_pipelines(self) -> None:
        """Update the available models from Vertex AI"""
        if not self.access_token:
            self.pipelines = []
            return

        try:
            endpoint = f"https://{self.valves.REGION}-aiplatform.googleapis.com/v1beta1/projects/{self.valves.GOOGLE_PROJECT_ID}/locations/{self.valves.REGION}/models"
            async with self.session.get(endpoint, headers=self._get_headers()) as response:
                response.raise_for_status()
                data = await response.json()
                models = data.get("models", [])
                self.pipelines = [
                    {
                        "id": model["name"],
                        "name": model["displayName"],
                        "version": model.get("version"),
                        "supportedPredictionTypes": model.get("supportedPredictionTypes", []),
                        "supportedDeploymentResourcesTypes": model.get("supportedDeploymentResourcesTypes", [])
                    }
                    for model in models
                ]
        except Exception as e:
            print(f"Error updating pipelines: {e}")
            self.pipelines = [
                {
                    "id": "error",
                    "name": f"Could not fetch models from Vertex AI: {e}",
                }
            ]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def pipe(
        self, 
        user_message: str, 
        model_id: Optional[str] = None, 
        messages: List[dict] = None, 
        body: dict = None
    ) -> Union[str, AsyncIterator[str]]:
        if not self.access_token:
            raise ValueError("Error: Service account not authenticated")

        model_id = model_id or self.valves.DEFAULT_MODEL
        if not model_id:
            raise ValueError("Error: No model specified and no default model set")

        try:
            # Determine the appropriate endpoint based on the model type
            if "text-generation" in model_id.lower():
                endpoint = f"https://{self.valves.REGION}-aiplatform.googleapis.com/v1beta1/{model_id}:generateContent"
            elif "chat" in model_id.lower():
                endpoint = f"https://{self.valves.REGION}-aiplatform.googleapis.com/v1beta1/{model_id}:streamGenerateContent"
            else:
                endpoint = f"https://{self.valves.REGION}-aiplatform.googleapis.com/v1beta1/{model_id}:predict"

            payload = self._prepare_payload(user_message, messages, body)

            async with self.session.post(endpoint, headers=self._get_headers(), json=payload) as response:
                if response.status == 200:
                    if "stream" in endpoint:
                        return self._stream_response(response)
                    else:
                        data = await response.json()
                        return self._process_response(data)
                else:
                    await self._handle_error_response(response)
        except aiohttp.ClientError as e:
            raise Exception(f"Network error: {str(e)}")
        except Exception as e:
            raise Exception(f"An error occurred: {str(e)}")

    async def count_tokens(self, text: str) -> int:
        endpoint = f"https://{self.valves.REGION}-aiplatform.googleapis.com/v1beta1/{self.valves.DEFAULT_MODEL}:countTokens"
        payload = {"content": text}
        async with self.session.post(endpoint, headers=self._get_headers(), json=payload) as response:
            if response.status == 200:
                data = await response.json()
                return data.get("totalTokens", 0)
            else:
                await self._handle_error_response(response)

    async def evaluate_model(self, model_id: str, evaluation_data: List[dict]) -> dict:
        endpoint = f"https://{self.valves.REGION}-aiplatform.googleapis.com/v1beta1/{model_id}:evaluateModel"
        payload = {"instances": evaluation_data}
        async with self.session.post(endpoint, headers=self._get_headers(), json=payload) as response:
            if response.status == 200:
                return await response.json()
            else:
                await self._handle_error_response(response)

    async def export_model(self, model_id: str, output_config: dict) -> str:
        endpoint = f"https://{self.valves.REGION}-aiplatform.googleapis.com/v1beta1/{model_id}:export"
        payload = {"outputConfig": output_config}
        async with self.session.post(endpoint, headers=self._get_headers(), json=payload) as response:
            if response.status == 200:
                data = await response.json()
                operation_id = data.get("name")
                self.operations[operation_id] = "export_model"
                return operation_id
            else:
                await self._handle_error_response(response)

    async def check_operation_status(self, operation_id: str) -> dict:
        endpoint = f"https://{self.valves.REGION}-aiplatform.googleapis.com/v1beta1/{operation_id}"
        async with self.session.get(endpoint, headers=self._get_headers()) as response:
            if response.status == 200:
                return await response.json()
            else:
                await self._handle_error_response(response)

    def get_models(self) -> List[dict]:
        return self.pipelines

    def _get_headers(self):
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }

    def _prepare_payload(self, user_message: str, messages: List[dict], body: dict) -> dict:
        instances = [{"content": msg["content"]} for msg in (messages or []) if msg["role"] != "system"]
        instances.append({"content": user_message})

        parameters = {
            "temperature": body.get("temperature", 0.7),
            "topP": body.get("top_p", 0.95),
            "topK": body.get("top_k", 40),
            "maxOutputTokens": body.get("max_tokens", self.valves.MAX_OUTPUT_TOKENS),
        }
        if body.get("stop"):
            parameters["stopSequences"] = body["stop"]

        if self.valves.USE_PERMISSIVE_SAFETY:
            parameters["safetySettings"] = [
                {"category": "HARM_CATEGORY_DEROGATORY", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_TOXICITY", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_VIOLENCE", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUAL", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_MEDICAL", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS", "threshold": "BLOCK_NONE"}
            ]

        return {
            "instances": instances,
            "parameters": parameters
        }

    async def _stream_response(self, response: aiohttp.ClientResponse) -> AsyncIterator[str]:
        async for chunk in response.content.iter_any():
            yield self._process_chunk(chunk)

    def _process_response(self, data: dict) -> str:
        predictions = data.get("predictions", [])
        if predictions:
            return predictions[0].get("content", "No content generated")
        return "No predictions generated"

    def _process_chunk(self, chunk: bytes) -> str:
        # Process the chunk and extract the generated content
        # This is a placeholder implementation and may need to be adjusted based on the actual response format
        try:
            data = json.loads(chunk.decode())
            return data.get("outputs", [{}])[0].get("content", "")
        except json.JSONDecodeError:
            return ""

    async def _handle_error_response(self, response: aiohttp.ClientResponse):
        try:
            error_data = await response.json()
            error_message = error_data.get("error", {}).get("message", "Unknown error")
        except:
            error_message = await response.text()
        raise Exception(f"API Error: {response.status} - {error_message}")