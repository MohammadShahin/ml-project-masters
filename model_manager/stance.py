from src.stance_ML.model_handler import predict_stance


def DetectStance(arguments, topic, debug=False):
    for arg in arguments:
        print(arg)
    result = predict_stance(arguments, topic)

    for arg, r in zip(arguments, result):
        arg['stance'] = r
        if debug:
            print(arg)
            print(r)

    return arguments
