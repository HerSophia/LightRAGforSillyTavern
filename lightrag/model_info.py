# model_info.py

import httpx
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from typing import List
from dataclasses import dataclass, field
import os


@dataclass
class ModelInfoManager:
    api_key: str
    base_url: str
    available_models: List[str] = field(default_factory=list)

    async def fetch_models(self):
        """
        Fetches the list of available models from the API and updates the `available_models` attribute.
        """
        headers = {"Authorization": f"Bearer {self.api_key}"}

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(f"{self.base_url}/models", headers=headers)
                response.raise_for_status()

                # Parse response to extract model IDs
                response_data = response.json()
                self.available_models = [model["id"] for model in response_data.get("data", [])]

                return self.available_models

            except httpx.HTTPStatusError as e:
                raise HTTPException(
                    status_code=e.response.status_code,
                    detail=f"Failed to fetch models: {e.response.text}",
                )
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"An unexpected error occurred: {str(e)}"
                )

    async def get_model_info(self):
        """
        Returns the models as a FastAPI JSONResponse.
        """
        try:
            await self.fetch_models()
            return JSONResponse(
                content={"available_models": self.available_models},
                status_code=200,
            )
        except HTTPException as e:
            raise e
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to get model information: {str(e)}"
            )
