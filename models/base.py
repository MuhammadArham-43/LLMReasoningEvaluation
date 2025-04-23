from typing import Optional, Tuple, Dict
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

    def __call__(self, system_prompt: str, prompt: str, **kwargs) -> str:
        response = self.generate(system_prompt=system_prompt, prompt=prompt, **kwargs)
        return self.postprocess(response)
    
    @abstractmethod
    def generate(self, system_prompt: str, prompt: str) -> Tuple[str, Dict]:
        pass

    def postprocess(self, response: str) -> str:
        return response



class CohereLLM(BaseLLM):

    def load_model(self, model_name: str):
        import cohere
        self.model_name = model_name
        self.client = cohere.ClientV2(api_key=self.api_key)
    
    def generate(self, system_prompt: str, prompt: str) -> Tuple[str, Dict]:
        res = self.client.chat(
            model=self.model_name,
            temperature=1.0,
            p=1.0,
            max_tokens=2049,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        usage = {
            "billed_units": {
                "input_tokens": res.usage.billed_units.input_tokens,
                "output_tokens": res.usage.billed_units.output_tokens
            },
            "tokens": {
                "input_tokens": res.usage.tokens.input_tokens,
                "output_tokens": res.usage.tokens.output_tokens
            }
        }
        return res.message.content[0].text, usage