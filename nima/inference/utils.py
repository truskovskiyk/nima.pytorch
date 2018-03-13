def format_output(mean_score, std_score, prob):
    return {
        'mean_score': float(mean_score),
        'std_score': float(std_score),
        'scores': [float(x) for x in prob]
    }
