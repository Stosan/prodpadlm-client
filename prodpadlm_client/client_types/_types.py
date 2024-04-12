from typing import Literal, Optional, Union

from pydantic import BaseModel

from .messages import Message

class TextBlock(BaseModel):
    text: str

    type: Literal["text"]

ContentBlock = TextBlock

class MessageStart(BaseModel):
    type: str = "message_start"
    message: dict


class ContentBlockStart(BaseModel):
    type: str = "content_block_start"
    index: int
    content_block: dict


class Ping(BaseModel):
    type: str = "ping"


class TextDelta(BaseModel):
    type: str = "text_delta"
    text: str


class ContentBlockDelta(BaseModel):
    type: str = "content_block_delta"
    index: int
    delta: TextDelta


class ContentBlockStop(BaseModel):
    type: str = "content_block_stop"
    index: int


class MessageDelta(BaseModel):
    type: str = "message_delta"
    delta: dict


class MessageStop(BaseModel):
    type: str = "message_stop"


MessageStreamEvent = Union[
    MessageStart,
    ContentBlockStart,
    Ping,
    ContentBlockDelta,
    ContentBlockStop,
    MessageDelta,
    MessageStop,
]
