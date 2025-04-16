from prompts import get_prompt_templates_for_dataset
from models import CohereLLM
from dataloader import GSM8K
import os
import time
import json
from tqdm import tqdm

if __name__ == "__main__":
    """
    TODO: Convert parameters to YAML config for easier setup.
    """
    
    RESULT_DIR = "results"
    MODEL_NAME = "command-a-03-2025"
    model = CohereLLM("command-a-03-2025", api_key="")
    data = GSM8K()
    prompt_templates = get_prompt_templates_for_dataset(
        dataset_name=data.get_dataset_name().lower()
    )

    for prompt_type, template_str in prompt_templates.items():
        save_dir = os.path.join(RESULT_DIR, data.get_dataset_name(), MODEL_NAME)
        os.makedirs(save_dir, exist_ok=True)
        responses = []
        for iteration, example in enumerate(data):
            prompt = template_str.format(example['question'], data.get_data())
            start_time = time.time()
            response = model(prompt=prompt)
            end_time = time.time()
            
            obj = {
                "question": example["question"],
                "ground_truth_answer": example["answer"],
                "prompt_template": prompt,
                "response": response,
                "response_time": end_time - start_time
            }
            responses.append(obj)

            if (iteration + 1) % 5 == 0:
                with open(
                    os.path.join(save_dir, prompt_type + ".json"), 
                    "w", 
                    encoding="utf-8"
                ) as f:
                    json.dump(responses, f, ensure_ascii=False, indent=4)
                    break