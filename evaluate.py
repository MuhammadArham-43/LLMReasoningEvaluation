import json
from evaluation.math_evals.eval import evaluate_gsm8k


if __name__ == "__main__":
    RESPONSE_FILE_PATH = "/Users/muhammadarham/Drive/LLMReasoningEvaluation/results/GSM8K/command-r-08-2024/zero-shot-cot.json"
    with open(RESPONSE_FILE_PATH, "r") as _f:
        responses = json.load(_f)

    correct_responses = 0
    total_responses = 0
    for item in responses:
        gt_answer = item["ground_truth_answer"]
        model_res = item["response"]
        correct, total = evaluate_gsm8k(gt_answer, model_res)
        correct_responses += correct
        total_responses += total

    print("=" * 80)
    print(f"Total Responses Generated: {len(responses)}")
    print(f"Total Responses Considered: {total_responses}")
    print(f"Correct Responses: {correct_responses}")
    print(f"GSM8K ACCURACY: {round(correct_responses/total_responses, 3)}")