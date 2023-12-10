import random

# from src.evaluator_ML.model_handler import prediction_handler


def GetScores(input_sentences):
    # return prediction_handler(input_sentences)
    scores = []
    for sent in input_sentences:
        scores.append(random.random())
    return scores
