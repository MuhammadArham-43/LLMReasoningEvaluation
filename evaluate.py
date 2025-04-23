import re
import json

def evaluate_exact_match(gt_answer, model_response):
    gt = re.search(r'\n####\s*(.*)', gt_answer).group(1).strip()
    model_res = re.search(r'\\boxed\{(.*?)\}', model_response).group(1).strip()
    try:
        gt_float = float(gt.replace(",", "").replace("%", ""))
        model_res_float = float(model_res.replace(",", "").replace("%", "").replace("\\", "").replace("{", "").replace("}", ""))
    except Exception as e:
        print(gt, model_res)
        print(e)
        return False, 0

    if gt_float != model_res_float:
        print(gt_float, model_res_float)
    return 1 if gt == model_res else 0, 1


if __name__ == "__main__":
    RESPONSE_FILE_PATH = "results/GSM8K/command-a-03-2025/zero-shot-direct-answer.json"
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