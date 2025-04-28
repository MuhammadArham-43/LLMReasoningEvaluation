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
    
    API_KEY = os.getenv("COHERE_API_KEY")
    assert API_KEY and API_KEY != "", "MODEL API KEY IS REQUIRED"
    
    RESULT_DIR = "results"
    MODEL_NAME = "command-r-08-2024"
    model = CohereLLM(MODEL_NAME, api_key=API_KEY)
    data = GSM8K()
    prompt_templates = get_prompt_templates_for_dataset(
        dataset_name=data.get_dataset_name().lower()
    )

    for prompt_type, prompt_template in prompt_templates.items():
        print(f"PROMPT TYPE: {prompt_type}")
        save_dir = os.path.join(RESULT_DIR, data.get_dataset_name(), MODEL_NAME)
        output_file_path = os.path.join(save_dir, prompt_type + ".json")
        responses = []
        if os.path.exists(output_file_path):
            with open(output_file_path, "r") as _f:
                responses = json.load(_f)
        start_idx = len(responses) 
        os.makedirs(save_dir, exist_ok=True)
        iteration = 0
        for example in tqdm(data, desc=prompt_type):
            if iteration < start_idx:
                iteration += 1
                continue
            system_prompt, prompt = prompt_template.format(question=example['question'].strip(), data=data.get_train_data())
            start_time = time.time()
            response, usage = model(system_prompt=system_prompt, prompt=prompt)
            end_time = time.time()
            
            obj = {
                "question": example["question"],
                "ground_truth_answer": example["answer"],
                "system_prompt": system_prompt,
                "prompt_template": prompt,
                "response": response,
                "response_time": end_time - start_time,
                "usage": usage
            }
            responses.append(obj)

            if (iteration + 1) % 5 == 0:
                with open(output_file_path, "w", encoding="utf-8") as f:
                    json.dump(responses, f, ensure_ascii=False, indent=4)
            iteration += 1

        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(responses, f, ensure_ascii=False, indent=4)