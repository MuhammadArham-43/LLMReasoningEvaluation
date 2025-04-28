import re
import json

def extract_boxed_content(text):
    """
    Extract content inside the LAST \\boxed{...}, properly handling nested braces.
    """
    matches = list(re.finditer(r'\\boxed\{', text))
    if not matches:
        return None

    # Take the last match
    match = matches[-1]
    start = match.end()
    brace_count = 1
    i = start

    while i < len(text):
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                return text[start:i]
        i += 1

    return None  # if no matching closing brace found

def extract_numeric(text):
    """
    Extract the first numeric value allowing commas inside numbers.
    """
    numeric_match = re.search(r'[-+]?\d[\d,]*\.?\d*', text)
    if not numeric_match:
        return None
    return numeric_match.group()


def evaluate_exact_match(gt_answer, model_response):
    gt = re.search(r'\n####\s*(.*)', gt_answer).group(1).strip()
    model_boxed_content = extract_boxed_content(model_response)
    if model_boxed_content is None:
        return False, 0
    model_res_numeric = extract_numeric(model_boxed_content)
    if model_res_numeric is None:
        print("No numeric value found in model response:", model_boxed_content)
        return False, 1
    # print("TEXT CONTENT:", model_boxed_content, "NUMERIC MATCH:", model_res_numeric, "GROUND TRUTH:", gt)
    try:
        gt_float = float(gt.replace(",", "").replace("%", ""))
        model_res_float = float(model_res_numeric.replace(",", "").strip())
    except Exception as e:
        print(gt, model_res)
        print(e)
        return False, 0

    if gt_float != model_res_float:
        print(gt_float, model_res_float)
    return 1 if gt_float == model_res_float else 0, 1


if __name__ == "__main__":
    RESPONSE_FILE_PATH = "results/GSM8K/command-r-08-2024/analogical-prompting.json"
    with open(RESPONSE_FILE_PATH, "r") as _f:
        responses = json.load(_f)

    correct_responses = 0
    total_responses = 0
    for item in responses:
        gt_answer = item["ground_truth_answer"]
        model_res = item["response"]
        correct, total = evaluate_exact_match(gt_answer, model_res)
        correct_responses += correct
        total_responses += total

    print("=" * 80)
    print(f"Total Responses Generated: {len(responses)}")
    print(f"Total Responses Considered: {total_responses}")
    print(f"Correct Responses: {correct_responses}")
    print(f"GSM8K ACCURACY: {round(correct_responses/total_responses, 3)}")