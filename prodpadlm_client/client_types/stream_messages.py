from typing import Iterator, List,TYPE_CHECKING
from prodpadlm_client.client_types._types import *
from typing_extensions import assert_never
from pydantic import Field



class MessageStream:
    text_stream: List[MessageStreamEvent] = Field(default_factory=list)
    
    def __init__(self,stream: MessageStreamEvent):
        self.text_stream = list(self.__stream_text__(self.parse_data(stream)))  # Convert generator to list
   
        
    def __stream_text__(self,stream_data) -> Iterator[str]:
        print(stream_data)
        if stream_data.type == "content_block_delta" and stream_data.delta.type == "text_delta":
            yield stream_data.delta.text
            
    def parse_data(self,data: dict) -> MessageStreamEvent:
        event_type = data.get('type')
        if event_type == "message_start":
            return MessageStart(**data)
        elif event_type == "content_block_start":
            return ContentBlockStart(**data)
        elif event_type == "ping":
            return Ping(**data)
        elif event_type == "text_delta":
            return TextDelta(**data)
        elif event_type == "content_block_delta":
            return ContentBlockDelta(**data)
        elif event_type == "content_block_stop":
            return ContentBlockStop(**data)
        elif event_type == "message_delta":
            return MessageDelta(**data)
        elif event_type == "message_stop":
            return MessageStop(**data)
        else:
            # we only want exhaustive checking for linters, not at runtime
            if TYPE_CHECKING:  # type: ignore[unreachable]
                assert_never(event_type)


class MessageStreamManager(MessageStream):
    def __init__(self, data):
        super().__init__(data)
        #self.generator = self.__stream_text__(self.parse_data(data))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        #self.generator.close()
        pass

