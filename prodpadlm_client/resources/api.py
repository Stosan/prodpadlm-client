import json
from typing import Iterable, List
import httpx
from typing_extensions import Literal, Required, TypedDict

from prodpadlm_client.client_types._types import *

from prodpadlm_client.client_types.messages import Message
from prodpadlm_client.client_types.stream_messages import MessageStreamManager

# default timeout is 10 minutes
DEFAULT_TIMEOUT = httpx.Timeout(timeout=600.0, connect=5.0)
DEFAULT_MAX_RETRIES = 2
DEFAULT_CONNECTION_LIMITS = httpx.Limits(
    max_connections=1000, max_keepalive_connections=100
)

__all__ = ["MessageParam"]


def parse_concatenated_json(string):
    try:
        # Attempt to load the string as is (useful if the string is already a valid JSON object or array)
        return json.loads(string)
    except json.JSONDecodeError:
        # If it fails, attempt to find individual JSON objects
        objects = []
        remaining = string.strip()
        while remaining:
            try:
                obj, idx = json.JSONDecoder().raw_decode(remaining)
                objects.append(obj)
                remaining = remaining[idx:].strip()
            except json.JSONDecodeError as e:
                # Unable to parse the remaining string as JSON, break the loop
                print(f"Error parsing JSON: {e}")
                break
        return objects

class MessageParam(TypedDict, total=False):
    content: str

    role: Required[Literal["user", "assistant"]]


class ProdPADLM_API:
        
    class Client:
        def __init__(self, api_key: str, base_url: str, default_headers: str = ""):
            headers = {"Content-Type": "application/json", "X-API-Key": api_key}
            self._post = httpx.Client(headers=headers)
            self.url = base_url

        def create(
            self,
            *,
            max_tokens: int,
            messages: Iterable[MessageParam],
            model: str = "",
            stop_sequences: List[str] = "",
            system: str = "",
            temperature: float = 0.7,
            top_k: int = 0,
            top_p: float = 0,
            stream: bool = False,
        ) -> Message:
            response = self._post.post(
                self.url + "/api/v1/generate",
                json={
                    "max_tokens": max_tokens,
                    "messages": messages,
                    "model": model,
                    "stop_sequences": stop_sequences,
                    "stream": stream,
                    "system": system,
                    "temperature": temperature,
                    "top_k": top_k,
                    "top_p": top_p,
                },
                timeout= DEFAULT_TIMEOUT
            )
            resp = Message(**response.json())
            return resp
        

        def stream(
            self,
            *,
            max_tokens: int,
            messages: Iterable[MessageParam],
            model: str = "",
            stop_sequences: List[str] = "",
            system: str = "",
            temperature: float = 0.7,
            top_k: int = 0,
            top_p: float = 0,
        ) -> Message:
            with self._post.stream("POST", self.url + "/api/v1/generate",
                                    json={
                                    "max_tokens": max_tokens,
                                    "messages": messages,
                                    "model": model,
                                    "stop_sequences": stop_sequences,
                                    "stream": True,
                                    "system": system,
                                    "temperature": temperature,
                                    "top_k": top_k,
                                    "top_p": top_p,

                                }) as response:
                    for data in response.iter_lines():
                        if data:
                            # Parse the concatenated JSON string
                            parsed_objects = parse_concatenated_json(data)

                            # Do something with the parsed objects (for demonstration, just print them)
                            for obj in parsed_objects:
                                with MessageStreamManager(obj["data"]) as msg:
                                    yield msg
                    
          
         
        
        
    class AsyncClient:
        def __init__(self, api_key: str, base_url: str, default_headers: str = ""):
            headers = {"Content-Type": "application/json", "X-API-Key": api_key}
            self._post =  httpx.AsyncClient(headers=headers, timeout=DEFAULT_TIMEOUT)
            self.url = base_url

        async def create(
            self,
            *,
            max_tokens: int,
            messages: Iterable[MessageParam],
            model: str = "",
            stop_sequences: List[str] = "",
            system: str = "",
            temperature: float = 0.7,
            top_k: int = 0,
            top_p: float = 0,
            stream: bool = False,
        ) -> Message:
            async with self._post as client:
                response = await client.post(
                self.url + "/api/v1/generate",
                json={
                    "max_tokens": max_tokens,
                    "messages": messages,
                    "model": model,
                    "stop_sequences": stop_sequences,
                    "stream": stream,
                    "system": system,
                    "temperature": temperature,
                    "top_k": top_k,
                    "top_p": top_p,
                },
            )
            resp=json.loads(response.read())
            parsed_resp = Message(**resp)
            return parsed_resp

        async def stream(
            self,
            *,
            max_tokens: int,
            messages: Iterable[MessageParam],
            model: str = "",
            stop_sequences: List[str] = "",
            system: str = "",
            temperature: float = 0.7,
            top_k: int = 0,
            top_p: float = 0,
        ) -> Message:
            async with self._post as client:
                with client.stream("POST", self.url + "/api/v1/generate",
                                    json={
                                    "max_tokens": max_tokens,
                                    "messages": messages,
                                    "model": model,
                                    "stop_sequences": stop_sequences,
                                    "stream": True,
                                    "system": system,
                                    "temperature": temperature,
                                    "top_k": top_k,
                                    "top_p": top_p,

                                }) as response:
                    async for data in response.aiter_lines():
                        if data:
                            # Parse the concatenated JSON string
                            parsed_objects = parse_concatenated_json(data)

                            # Do something with the parsed objects (for demonstration, just print them)
                            for obj in parsed_objects:
                                with MessageStreamManager(obj["data"]) as msg:
                                    yield msg