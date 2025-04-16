from typing import Optional, List, Dict
from abc import ABC, abstractmethod
import jinja2
import time

class BasePromptTemplate(ABC):

    def __init__(self, template_str: Optional[str] = None, template_path: Optional[str] = None) -> None:
        super().__init__()
        if template_str:
            self.template_str = template_str
        elif template_path:
            with open(template_path, "r") as _f:
                self.template_str = _f.read()
        else:
            raise ValueError("Either template_str or template_path must be provided.")

        self.template = jinja2.Template(self.template_str)

    def format(self, question: str, data = None, **kwargs):
        return self.template.render(question=question, **kwargs)


class ZeroShotTemplate(BasePromptTemplate):
    pass

class FewShotTemplate(BasePromptTemplate):

    def _generate_fewshot_examples(self, data, num_examples: int = 3, shuffle: bool = True):
        if shuffle:
            return data.shuffle().select(range(num_examples))
        return data.select(range(num_examples))

    def format(self, question: str, data, **kwargs):
        examples = self._generate_fewshot_examples(data)
        return self.template.render(examples=examples, question=question, **kwargs)

class ChainOfThoughtTemplate(BasePromptTemplate):
    pass