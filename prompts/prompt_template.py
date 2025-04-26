from typing import Optional, List, Dict, Tuple
from abc import ABC, abstractmethod
import jinja2
import time

class BasePromptTemplate(ABC):

    def __init__(self, template_dict: Dict) -> None:
        super().__init__()
        self.template_dict = template_dict
        self.system_prompt = jinja2.Template(self.template_dict['system_prompt'])
        self.prompt = jinja2.Template(self.template_dict['prompt'])

    def format(self, question: str, **kwargs) -> Tuple[str, str]:
        return self.system_prompt.render(**kwargs), self.prompt.render(prompt=question, **kwargs)

    def get_system_prompt_template(self) -> str:
        return self.system_prompt if self.system_prompt else None
    
    def get_prompt_template(self) -> str:
        return self.prompt


class ZeroShotTemplate(BasePromptTemplate):
    pass

class FewShotTemplate(BasePromptTemplate):

    def _generate_fewshot_examples(self, data, num_examples: int = 3, shuffle: bool = True):
        if shuffle:
            return data.shuffle().select(range(num_examples))
        return data.select(range(num_examples))

    def format(self, question: str, data, **kwargs):
        examples = self._generate_fewshot_examples(data, num_examples=8)
        return self.system_prompt.render(**kwargs), self.prompt.render(examples=examples, prompt=question.strip(), **kwargs)

class ChainOfThoughtTemplate(BasePromptTemplate):
    pass

class SelfCriticism(BasePromptTemplate):
    pass

class DecompositionPrompting(BasePromptTemplate):
    pass