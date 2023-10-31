from src.transformers_ML.config import *
from src.transformers_ML.utils import *
from src.transformers_ML.data_loader import DataLoadHandler
from src.transformers_ML.model_handler import *
from src.data_parser.utils import ParseEssays, ParseWebDiscourse, ProcessExampleEssay

from sklearn.linear_model import LogisticRegression

# svm dependencies
from src.classical_ML.model_handler import * 
from src.classical_ML.src.data_reader import ProcessRowSentences

# Set the seed value all over the place to make this reproducible.
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

def argNonArgMap(y):
    tmp = []
    for c in y:
        if c == 'c':
            tmp.append(1)
        elif c == 'p':
            tmp.append(2)
        else:
            tmp.append(0)
    return tmp

def GetTrainTestLists(x, y, fold=5):
    x_train_list, x_test_list, y_train_list, y_test_list = [], [], [], []

    test_data_length = len(x) // fold
    for index in range(fold):

        if (index + 1) * test_data_length <= len(x):
            x_test = x[index * test_data_length : (index + 1) * test_data_length]
            y_test = y[index * test_data_length : (index + 1) * test_data_length]
            x_train = x[ : index * test_data_length] + x[(index + 1) * test_data_length : ]
            y_train = y[ : index * test_data_length] + y[(index + 1) * test_data_length : ]
        else:
            x_test = x[index * test_data_length : ]
            y_test = y[index * test_data_length : ]
            x_train = x[ : index * test_data_length]
            y_train = y[ : index * test_data_length]

        x_train_list.append(x_train)
        x_test_list.append(x_test)
        y_train_list.append(y_train)
        y_test_list.append(y_test)
        print(len(x_train), len(x_test), len(y_train), len(y_test))

    return x_train_list, x_test_list, y_train_list, y_test_list


def StackSvmBert():
    data_dir = ARG_EXTRACTION_ROOT_DIR + '/corpora/parsed-corpora/'

    with open(data_dir + 'essays_sentences.json', encoding='utf-8') as f:
        essays_sentences = json.load(f)
        y_essays = argNonArgMap([sent['sent-class'] for sent in essays_sentences])

    with open(data_dir + 'web_discourse.json', encoding='utf-8') as f:
        web_d_sentences = json.load(f)
        y_web_d = argNonArgMap([sent['sent-class'] for sent in web_d_sentences])
        web_d_sentences = ProcessRowSentences([sent['sent-text'] for sent in web_d_sentences])

    sentences_all = essays_sentences + web_d_sentences
    y = y_essays + y_web_d

    temp_text, test_text, temp_labels, test_labels = train_test_split(sentences_all, y,
                                                                            random_state = 2018,
                                                                            shuffle = True,
                                                                            test_size = 0.20,
                                                                            stratify = y)

    print('training bert and svm ...')
    bert_model, bert_preds = bert_train_and_predict(temp_text, temp_labels, test_text)
    svm_model, svm_preds = svm_train_and_predict(temp_text, temp_labels, test_text)

    meta_fold = 5
    x_train_list, x_test_list, y_train_list, y_test_list = GetTrainTestLists(temp_text, temp_labels, meta_fold)
    
    x_meta_train, y_meta_train = [], []
    for index in range(meta_fold):
        print(f'meta model data processing: fold {index + 1}')

        train_text, val_text, train_labels, val_labels = x_train_list[index], x_test_list[index], y_train_list[index], y_test_list[index]
        print(len(train_text), len(val_text), len(train_labels), len(val_labels))

        bert_model, y_bert = bert_train_and_predict(train_text, train_labels, val_text)
        svm_model, y_svm = svm_train_and_predict(train_text, train_labels, val_text)
        x_meta_train += [y_svm[idx][:1] + y_bert[idx] for idx in range(len(val_text))]
        y_meta_train += val_labels
        print(len(x_meta_train), len(y_meta_train))
    
    print('finish meta train data preparation')
    meta_model = LogisticRegression()
    
    if not os.path.isfile('lr_model_3cs.sav'):    
        meta_model.fit(x_meta_train, y_meta_train)

        # Save the model to disk
        filename = 'lr_model_3cs.sav'
        pickle.dump(meta_model, open(filename, 'wb'))
        print("Model saved successfully.")
    else:
        # Load the model from disk
        meta_model = pickle.load(open('lr_model_3cs.sav', 'rb'))
        print("Model loaded successfully.")
        
    # testing the performance of the whole pipline
    x_meta_test = [svm_preds[idx][:1] + bert_preds[idx] for idx in range(len(test_text))]
    meta_preds = meta_model.predict(x_meta_test)

    print(classification_report(test_labels, meta_preds, digits=4))


def TestBertSvm():
    data_dir = ARG_EXTRACTION_ROOT_DIR + '/corpora/parsed-corpora/'

    with open(data_dir + 'essays_sentences.json', encoding='utf-8') as f:
        essays_sentences = json.load(f)
        y_essays = argNonArgMap([sent['sent-class'] for sent in essays_sentences])

    with open(data_dir + 'web_discourse.json', encoding='utf-8') as f:
        web_d_sentences = json.load(f)
        y_web_d = argNonArgMap([sent['sent-class'] for sent in web_d_sentences])
        web_d_sentences = ProcessRowSentences([sent['sent-text'] for sent in web_d_sentences])

    sentences_all = essays_sentences + web_d_sentences
    y = y_essays + y_web_d

    temp_text, test_text, temp_labels, test_labels = train_test_split(sentences_all, y,
                                                                            random_state = 2018,
                                                                            shuffle = True,
                                                                            test_size = 0.20,
                                                                            stratify = y)

    print('testing bert and svm ...')
    bert_preds = bert_test_first_stage(test_text)
    svm_preds = svm_test_second_stage(test_text)
    
    # testing the performance of the whole pipline
    y_pred = []
    for pred_1, pred_2 in zip(bert_preds, svm_preds):
        if pred_1 == 0:
            y_pred.append(0)
        else:
            if pred_2 == 0:
                y_pred.append(1)
            else:
                y_pred.append(2)
    
    print(classification_report(test_labels, y_pred, digits=4))
    

def TestSvmBert():
    data_dir = ARG_EXTRACTION_ROOT_DIR + '/corpora/parsed-corpora/'

    with open(data_dir + 'essays_sentences.json', encoding='utf-8') as f:
        essays_sentences = json.load(f)
        y_essays = argNonArgMap([sent['sent-class'] for sent in essays_sentences])

    with open(data_dir + 'web_discourse.json', encoding='utf-8') as f:
        web_d_sentences = json.load(f)
        y_web_d = argNonArgMap([sent['sent-class'] for sent in web_d_sentences])
        web_d_sentences = ProcessRowSentences([sent['sent-text'] for sent in web_d_sentences])

    sentences_all = essays_sentences + web_d_sentences
    y = y_essays + y_web_d

    temp_text, test_text, temp_labels, test_labels = train_test_split(sentences_all, y,
                                                                            random_state = 2018,
                                                                            shuffle = True,
                                                                            test_size = 0.20,
                                                                            stratify = y)

    print('testing bert and svm ...')
    svm_preds = svm_test_first_stage(test_text)
    bert_preds = bert_test_second_stage(test_text)

    # testing the performance of the whole pipline
    y_pred = []
    for pred_1, pred_2 in zip(svm_preds, bert_preds):
        if pred_1 == 0:
            y_pred.append(0)
        else:
            if pred_2 == 0:
                y_pred.append(1)
            else:
                y_pred.append(2)
    
    print(classification_report(test_labels, y_pred, digits=4))


def TestFinalModel():
    data_dir = ARG_EXTRACTION_ROOT_DIR + '/corpora/parsed-corpora/'
    sentences = ProcessExampleEssay()
    with open(data_dir + 'example_essay_sentences.json', encoding='utf-8') as f:
        essays_sentences = json.load(f)

    
    # print(sentences)
    # return 
    # test_text = [sent['sent-text'] for sent in essays_sentences]
    # for text in zip(test_text):
    #     print("Text:", text)
    #     print()
    # return
    print('getting the predictions ...')
    svm_preds = svm_test_first_stage(essays_sentences)
    bert_preds = bert_test_second_stage(essays_sentences)

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
    
    # printing the result
    for text, pred in zip(essays_sentences, y_pred):
        print("Text:", text['sent-text'])
        print("Prediction:", pred)
        print()
    # end the example


def TestAnnotationModel():
    data_dir = ARG_EXTRACTION_ROOT_DIR + '/corpora/parsed-corpora/'
    sentences = ProcessExampleEssay()
    with open(data_dir + 'example_essay_sentences.json', encoding='utf-8') as f:
        essays_sentences = json.load(f)


    # print(sentences)
    # return
    # test_text = [sent['sent-text'] for sent in essays_sentences]
    # for text in zip(test_text):
    #     print("Text:", text)
    #     print()
    # return
    print('getting the predictions ...')
    svm_preds = svm_test_first_stage(essays_sentences)
    bert_preds = bert_test_second_stage(essays_sentences)

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

    # printing the result
    for text, pred in zip(essays_sentences, y_pred):
        print("Text:", text['sent-text'])
        print("Prediction:", pred)
        print()
    # end the example


def TestModel():
    data_dir = ARG_EXTRACTION_ROOT_DIR + '/corpora/parsed-corpora/'

    with open(data_dir + 'ibm_premise_claim.json', encoding='utf-8') as f:
        essays_sentences = json.load(f)
        y_label = argNonArgMap([sent['sent-class'] for sent in essays_sentences])
        essays_sentences = ProcessRowSentences([sent['sent-text'] for sent in essays_sentences])


    # print(sentences)
    # return
    # test_text = [sent['sent-text'] for sent in essays_sentences]
    # for text in zip(test_text):
    #     print("Text:", text)
    #     print()
    # return
    print('getting the predictions ...')
    # svm_preds = svm_test_first_stage(essays_sentences)
    bert_preds = bert_test_second_stage(essays_sentences)

    # collecting the predictions
    # y_pred = []
    # for pred_1, pred_2 in zip(svm_preds, bert_preds):
    #     if pred_1 == 0:
    #         y_pred.append(0)
    #     else:
    #         if pred_2 == 0:
    #             y_pred.append(1)
    #         else:
    #             y_pred.append(2)

    y_pred = []
    cnt1 = 0
    cnt2 = 0
    for pred_1 in zip(bert_preds):
        if pred_1 == 0:
            y_pred.append(1)
            cnt1 = cnt1 + 1
        else:
            y_pred.append(2)
            cnt2 = cnt2 + 1

    print('claims: ', cnt1, 'premises', cnt2)
    # printing the result
    print(classification_report(y_label, y_pred))
    # end the example


def main(args):
    action, obj = args
    if 'train' == action:
        if 'bert' == obj:
            print('Running BERT training...')
            train_model(epochs=3)
        elif 'svm' == obj:
            print('Running SVM training...')
            train_svm()
        elif 'dbert' == obj:
            print('Running Double Bert training...')
            train_model_double(epochs=3)
        elif 'dbert2' == obj:
            print('Running Double Bert training snd Stage...')
            train_model_double_stage2(epochs=3)
        elif 'stack' == obj:
            StackSvmBert()

    elif 'parse' == action:
        if 'essays' == obj:
            print('Parsing Essays...')
            ParseEssays()

        elif 'webd' == obj:
            print('Parsing Web Discourse...')
            ParseWebDiscourse()
    elif 'test' == action:
        if 'svm' == obj:
            print('Testing SVM model...')
            test_double_svm()
        elif 'bert' == obj:
            print('Testing SVM model...')
            test_double_svm()
        elif 'bertsvm' == obj:
            print('Testing Bert->Svm model...')
            TestBertSvm()
        elif 'svmbert' == obj:
            print('Testing Svm->Bert model...')
            TestSvmBert()
        elif 'example' == obj:
            print('Testing Example...')
            TestAnnotationModel()
        elif 'ibm' == obj:
            print('Testing on IBM...')
            TestModel()
    else:
        print('wrong params has been given!')

if __name__ == '__main__':
    # main(sys.argv[1:])
    TestModel()
