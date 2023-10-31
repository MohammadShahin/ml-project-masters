import os
import time
import en_core_web_sm
import csv
import json

nlp = en_core_web_sm.load()

ARG_EXTRACTION_ROOT_DIR = os.path.abspath(os.getcwd())
DATASET_BASE_IN_DIR = ARG_EXTRACTION_ROOT_DIR + '/corpora/ibm-args/'
DATASET_BASE_OUT_DIR = ARG_EXTRACTION_ROOT_DIR + '/corpora/parsed-corpora/'


# total claims: 2257
# total premises: 3832

def _GetArgs():
    arguments = []
    with open(DATASET_BASE_IN_DIR + 'arg_quality_rank_30k.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                line_count += 1
                argument = {
                    'arg-text': row[0]
                    , 'arg-topic': row[1]
                    , 'arg-score-wa': float(row[3])
                    , 'arg-score-mace-p': float(row[4])
                    , 'arg-stance': int(row[5])
                    , 'arg-stance-conf': float(row[6])
                    , 'train': ('train' == row[2])
                }
                arguments.append(argument)
    return arguments


def ArgsUnification():
    print('start arguments processing ...')
    starting_time = time.time()
    args = _GetArgs()

    # save all sentences
    print('saving all arguments...')

    with open(DATASET_BASE_OUT_DIR + 'args_sentences.json', 'w', encoding='utf-8') as f:
        json.dump(args, f)


def LoadArgsSentences(file_name=DATASET_BASE_OUT_DIR + 'args_sentences.json'):
    print('Loading annotated arguments ...')
    try:
        with open(file_name, 'r', encoding='utf-8') as f:
            arguments_all = json.load(f)
    except Exception as e:
        print('Error: {}'.format(e))
        arguments_all = []

    return arguments_all
