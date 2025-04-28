import importlib
import argparse
from prompts import get_prompt_templates_for_dataset
import os
import time
import json
import yaml
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="config.yaml", help="Path to config file")
    return parser.parse_args()

def load_class_from_string(dotted_path):
    """
    Load a class from a dotted string path.
    Example: "dataloader.GSM8K" -> <class GSM8K>
    Example: "models.CohereLLM" -> <class CohereLLM>
    """
    module_path, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls



if __name__ == "__main__":
    args = parse_arguments()
    assert os.path.exists(args.config_file), "Ensure config yaml exists for experiment parameters"
    with open(args.config_file) as _f:
        config = yaml.safe_load(_f)

    ModelClass = load_class_from_string(config["model"]["class"])
    DatasetClass = load_class_from_string(config["dataset"]["class"])

    MODEL_NAME = config["model"]["name"]
    API_KEY = os.getenv(config["model"]["api_key_env_var"])
    assert API_KEY and API_KEY != "", "MODEL API KEY IS REQUIRED"

    model = ModelClass(model_name=MODEL_NAME, api_key=API_KEY)
    data = DatasetClass()

    prompt_templates = get_prompt_templates_for_dataset(
        dataset_name=data.get_dataset_name().lower()
    )
    prompts_to_use = config.get("prompts", {}).get("use", [])
    if prompts_to_use:
        prompt_templates = {k: v for k, v in prompt_templates.items() if k in prompts_to_use}

    RESULT_DIR = "results"
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

        progress_bar = tqdm(data, desc=prompt_type, total=len(data))
        for example in progress_bar:
            if iteration < start_idx:
                iteration += 1
                progress_bar.set_postfix(skipped=iteration)
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