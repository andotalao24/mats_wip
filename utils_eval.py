
def compare_answers(extracted_ans, correct_ans):
    """Compare extracted answer with correct answer."""
    if not extracted_ans or not correct_ans:
        return False
    extracted_ans = str(extracted_ans).split("$")[0].split("\n")[0]
    
    extracted_ans = str(extracted_ans).strip()
    correct_ans = str(correct_ans).strip()
    
    # Direct string comparison
    if extracted_ans == correct_ans:
        return True
    
    # Try numerical comparison
    try:
        extracted_float = float(extracted_ans)
        correct_float = float(correct_ans)
        if abs(extracted_float - correct_float) < 1e-6:
            return True
    except (ValueError, TypeError):
        pass
    
    # Try fraction comparison (e.g., "1/2" vs "0.5")
    try:
        if '/' in extracted_ans:
            num, denom = extracted_ans.split('/')
            extracted_float = float(num) / float(denom)
            correct_float = float(correct_ans)
            if abs(extracted_float - correct_float) < 1e-6:
                return True
        elif '/' in correct_ans:
            num, denom = correct_ans.split('/')
            correct_float = float(num) / float(denom)
            extracted_float = float(extracted_ans)
            if abs(extracted_float - correct_float) < 1e-6:
                return True
    except (ValueError, TypeError, ZeroDivisionError):
        pass
    return False