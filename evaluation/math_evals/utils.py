import re


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


def evaluate_exact_match(gt, model_boxed_content):
    if model_boxed_content is None:
        return 0, 0
    model_res_numeric = extract_numeric(model_boxed_content)
    if model_res_numeric is None:
        print("No numeric value found in model response:", model_boxed_content)
        return 0, 1
    # print("TEXT CONTENT:", model_boxed_content, "NUMERIC MATCH:", model_res_numeric, "GROUND TRUTH:", gt)
    try:
        gt_float = float(gt.replace(",", "").replace("%", ""))
        model_res_float = float(model_res_numeric.replace(",", "").strip())
    except Exception as e:
        print(e)
        return 0, 0

    if gt_float != model_res_float:
        print(gt_float, model_res_float)
    return (1,1) if gt_float == model_res_float else (0, 1)