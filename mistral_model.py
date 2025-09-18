from agno.models.base import Model, ModelResponse
from mistralai import Mistral

class MistralAgnoModel(Model):
    def __init__(self, api_key: str, model_id: str = "mistral-small"):
        super().__init__(id=model_id, provider="mistral")
        self.client = Mistral(api_key=api_key)

    def invoke(self, messages, **kwargs) -> ModelResponse:
        converted_messages = []
        for m in messages:
            if isinstance(m, dict):
                converted_messages.append(m)
            else:
                converted_messages.append({
                    "role": getattr(m, "role", "user"),
                    "content": getattr(m, "content", str(m))
                })

        response = self.client.chat.complete(
            model=self.id,
            messages=converted_messages
        )

        return ModelResponse(
            content=response.choices[0].message.content
        )

    async def ainvoke(self, messages, **kwargs) -> ModelResponse:
        return self.invoke(messages, **kwargs)

    def invoke_stream(self, messages, **kwargs):
        yield self.invoke(messages, **kwargs)

    async def ainvoke_stream(self, messages, **kwargs):
        yield self.invoke(messages, **kwargs)

    def parse_provider_response(self, response, **kwargs):
        return response

    def parse_provider_response_delta(self, delta, **kwargs):
        return delta
