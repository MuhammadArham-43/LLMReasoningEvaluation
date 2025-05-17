import re
from evaluation.math_evals.grader import grade_answer
from .utils import evaluate_exact_match, extract_boxed_content

def evaluate_gsm8k(gt, response):
    gt = re.search(r'\n####\s*(.*)', gt).group(1).strip()
    model_boxed_content = extract_boxed_content(response)
    response = evaluate_exact_match(gt, model_boxed_content)
    return response 


