from pathlib import Path
import os 
import yaml

from .prompt_template import BasePromptTemplate, ZeroShotTemplate, FewShotTemplate, ChainOfThoughtTemplate

PROMPT_TEMPLATES = {
    "zero-shot-direct-answer": ZeroShotTemplate,
    "zero-shot-cot": ZeroShotTemplate,
    "role-prompting": ZeroShotTemplate,
    "rereading-prompting": ZeroShotTemplate
}


def get_prompt_templates_for_dataset(dataset_name: str):
    templates_path = os.path.join(Path(__file__).parent, "templates", dataset_name + ".yaml")
    with open(templates_path, "r") as _f:
        template_config = yaml.safe_load(_f)
    
    templates = {}
    for strategy, template in template_config.items():
        template_cls = PROMPT_TEMPLATES.get(strategy, BasePromptTemplate)
        templates[strategy] = template_cls(template)
    return templates       
