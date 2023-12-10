import json
from itertools import groupby

from model_manager.matcher import GetSimilarity, GetRelevance
from model_manager.scorer import GetScores
from model_manager.stance import DetectStance
from src.classical_ML.model_handler import svm_test_first_stage
from src.transformers_ML.model_handler import bert_test_second_stage

from src.evaluator_ML.model_handler import prediction_handler


def GetAnnotations(input_sentences, debug=False):
    # essays_sentences = ProcessSingleEssay(input_sentences)
    print('getting the predictions ...')
    svm_preds = svm_test_first_stage(input_sentences)
    bert_preds = bert_test_second_stage(input_sentences)

    # collecting the predictions
    y_pred = []
    for pred_1, pred_2 in zip(svm_preds, bert_preds):
        if pred_1 == 0:
            y_pred.append('NON-ARG')
        else:
            if pred_2 == 0:
                y_pred.append('CLAIM')
            else:
                y_pred.append('PREMISE')

    if debug:
        for text, pred in zip(input_sentences, y_pred):
            print("Text:", text['sent-text'])
            print("Prediction:", pred)
            print()
    for text, pred in zip(input_sentences, y_pred):
        text['annotation'] = pred
    # end the example
    return input_sentences


def HollyHeuristic(input_sentences, topic):
    global premises, claims

    input_sentences = GetAnnotations(input_sentences, False)
    tree = []
    idx = 1
    last = topic
    # last['idx'] = idx
    # idx = idx + 1
    for sent in input_sentences:
        sent['idx'] = idx
        idx = idx + 1
    # paragraphs = input_sentences.group_by(input_sentences['parag-idx'])
    # Sort the list based on 'parag-idx'
    input_sentences.sort(key=lambda x: x['parag-idx'])

    # Group the elements in the list by 'parag-idx'
    paragraphs = groupby(input_sentences, key=lambda x: x['parag-idx'])

    claims = []
    # Iterate over the grouped elements
    for key, group in paragraphs:
        premises = []
        curr_claims = []

        for item in group:
            if item['annotation'] == 'PREMISE':
                premises.append(item['idx'])
            elif item['annotation'] == 'CLAIM':
                curr_claims.append(item)
        for claim in curr_claims:
            claim['premises'] = premises
        claims += curr_claims

    # detect the stance for each claim in the essay
    print('Stance ...')
    claims = DetectStance(claims, topic, debug=True)

    # match each claim with all premises in the same paragraph
    idx = 0
    for claim in claims:
        claim['claim_idx'] = idx
        idx = idx + 1

    print('Similarity ...')
    # calculate the similarity between claims
    similarity_matrix = GetSimilarity(claims, debug=True)

    print('Relevance ...')
    # calculate the relevance for each claim to the topic
    claims = GetRelevance(claims, topic, debug=True)

    print('Scoring ...')
    # calculate the argumentative effectiveness between claims and premises
    for sent in input_sentences:
        if sent['annotation'] == 'CLAIM':
            sentences_to_score = []
            for x in sent['premises']:
                sentences_to_score.append(input_sentences[x - 1]['sent-text']
                                          + ' [SEP] ' + sent['sent-text'])
            if len(sentences_to_score) > 0:
                scores = GetScores(sentences_to_score)
                # for s in scores:
                #     print(s[0])
                print("some EXTRA Debugging: ")
                for score, premise in zip(scores, sent['premises']):
                    input_sentences[premise - 1]['score'] = score
                    print(premise)
                    print("score : ", score)

                combined = list(zip(sent['premises'], scores))
                sorted_combined = sorted(combined, key=lambda x: x[1], reverse=True)
                first_three = sorted_combined[:4] if len(sorted_combined) >= 3 else sorted_combined
                sent['premises'], b = zip(*first_three)
                sent['premises'] = list(sent['premises'])
                sent['score'] = sum(b)
            else:
                sent['premises'] = []
                sent['score'] = 0.01

            print('text:', sent['sent-text'])
            print('score:', sent['score'])
            print('relevance:', sent['relevance'])
            print('premises:', sent['premises'])

        def my_map_function(value):
            return value.lower()

        # Apply the map function to the 'annotation' column
        sent['annotation'] = my_map_function(sent['annotation'])

    # print(len(claims))
    total_score = 0
    sorted_claims = sorted(claims, key=lambda x: x['score'] * x['relevance'], reverse=True)
    cluster = []
    for s in sorted_claims:
        print(s)
        drop = False
        for c in cluster:
            if similarity_matrix[s['claim_idx']][c['claim_idx']] >= 0.9:
                drop = True
                break
        if drop:
            continue
        total_score += s['score'] * s['relevance']
        s['score'] = s['score'] * s['relevance']
        cluster.append(s)

    # print('Cluster Claims ...')
    # for c in cluster:
    #     print(c)


    print('Total Score : ', total_score)
    print('-------------------------------------')

    return input_sentences, total_score


























    # total_score = 3.450725172806113

def Holly_Heuristic(input_sentences, topic):
    file_path = "back_end/output_essays/gpt.json"
    sth , sth = HollyHeuristic(input_sentences, topic)

    with open(file_path, "r") as file:
        input_sentences = json.load(file)

    for i in range(len(input_sentences)):
        input_sentences[i]["sent-text"] = input_sentences[i].pop("text")

    claims = [sentence for sentence in input_sentences if sentence.get('annotation') == 'claim']
    print('Similarity ...')
    # calculate the similarity between claims
    similarity_matrix = GetSimilarity(claims, debug=True)

    total_score = 0
    sorted_claims = sorted(claims, key=lambda x: x['score'] * x['relevance'], reverse=True)
    cluster = []
    for s in sorted_claims:
        print(s)
        drop = False
        for c in cluster:
            if similarity_matrix[s['claim_idx']][c['claim_idx']] >= 0.9:
                drop = True
                break
        if drop:
            continue
        total_score += s['score'] * s['relevance']
        s['score'] = s['score'] * s['relevance']
        cluster.append(s)


    return input_sentences, total_score
