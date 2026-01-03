import math

def confidence_from_probs(probs):
    eps = 1e-9
    entropy = -sum(p * math.log(p + eps) for p in probs)
    max_entropy = math.log(len(probs))
    score = 1 - entropy / max_entropy
    if score > 0.75:
        return "высокая", round(score, 2)
    if score > 0.4:
        return "средняя", round(score, 2)
    return "низкая", round(score, 2)
