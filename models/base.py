from typing import Optional
from abc import ABC, abstractmethod


class BaseLLM(ABC):
    def __init__(self, model_name: str, api_key: Optional[str] = None, **kwargs) -> None:
        super().__init__()
        self.model_name = model_name
        self.api_key = api_key
        self.load_model(model_name=model_name, **kwargs)

    @abstractmethod
    def load_model(self, model_name: str):
        pass

    def __call__(self, prompt: str) -> str:
        response = self.generate(prompt)
        return self.postprocess(response)
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

    def postprocess(self, response: str) -> str:
        return response



class CohereLLM(BaseLLM):

    def load_model(self, model_name: str):
        import cohere
        self.client = cohere.ClientV2(api_key=self.api_key)
    
    def generate(self, prompt: str) -> str:
        res = self.client.chat(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return res.message.content[0].text


        


