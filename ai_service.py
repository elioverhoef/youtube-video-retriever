import os
import json
from typing import Optional, AsyncGenerator, List
import httpx
from dotenv import load_dotenv

load_dotenv()

class AIService:
    MODELS = [
        "google/gemini-exp-1206:free",
        "google/gemini-exp-1121:free",
        "google/gemini-exp-1114:free"
    ]
    
    def __init__(self):
        self.api_key = os.getenv("OPEN_ROUTER_API")
        if not self.api_key:
            raise ValueError("OPEN_ROUTER_API key not found in environment variables")
        
        self.base_url = "https://openrouter.ai/api/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    async def get_completion(self, 
                           prompt: str, 
                           system_prompt: Optional[str] = None,
                           model_index: int = 0) -> AsyncGenerator[str, None]:
        """
        Get streaming completion from the AI model with automatic fallback.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            model_index: Index of the model to use (0-2)
            
        Yields:
            Generated text chunks
        
        Raises:
            Exception: If all models fail
        """
        if model_index >= len(self.MODELS):
            raise Exception("All models exhausted")
            
        model = self.MODELS[model_index]
        
        payload = {
            "model": model,
            "messages": [
                *([{"role": "system", "content": system_prompt}] if system_prompt else []),
                {"role": "user", "content": prompt}
            ],
            "stream": True
        }
        
        try:
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=60.0
                ) as response:
                    if response.status_code == 429:  # Resource exhausted
                        # Try next model
                        async for chunk in self.get_completion(prompt, system_prompt, model_index + 1):
                            yield chunk
                        return
                        
                    response.raise_for_status()
                    
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data = json.loads(line[6:])
                            if content := data.get("choices", [{}])[0].get("delta", {}).get("content"):
                                yield content
                                
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:  # Resource exhausted
                # Try next model
                async for chunk in self.get_completion(prompt, system_prompt, model_index + 1):
                    yield chunk
            else:
                raise e 