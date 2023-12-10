from src.kp_analyzer_ML.KeyPointEvaluator import calc_similarity, calc_relevance


def GetRelevance(arguments, topic, debug=False):
    res = calc_relevance(arguments, topic)
    for arg, r in zip(arguments, res):
        arg['relevance'] = r
        if debug:
            print(arg['sent-text'])
            print(r)
            print()
    return arguments


def GetSimilarity(arguments, debug=False):
    matrix = calc_similarity(arguments)

    if debug:
        for i in range(len(arguments)):
            for j in range(len(arguments)):
                print(arguments[i]['sent-text'])
                print(arguments[j]['sent-text'])
                print('Similarity: ', matrix[i][j])
                print()
    return matrix
