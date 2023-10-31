from sentence_transformers import InputExample

from src.evaluator_ML.config import *

data_dir = ARG_EXTRACTION_ROOT_DIR + '/corpora/parsed-corpora/'
batch_size = 32
max_seq_len = 256


class DataLoadHandler:

    def __init__(self, test_size=0.2):
        arg_df = pd.read_json(data_dir + 'args_sentences.json')
        # arg_df = arg_df[arg_df['train']]
        # mean_scores = arg_df.groupby('arg-topic')['arg-score-mace-p'].mean()
        # arg_df = arg_df[arg_df['arg-topic'].isin(mean_scores[mean_scores <= 0.6].index)]

        print('Number of rows : ', len(arg_df))
        arg_df = arg_df[['train', 'arg-text', 'arg-topic', 'arg-score-mace-p']]
        # arg_df['arg-full'] = arg_df['arg-text'] + ' [SEP] ' + arg_df['arg-topic']
        # arg_df['arg-full'] = arg_df.apply(lambda row: row['arg-text'] + ' [SEP] ' + row['arg-topic'], axis=1)

        print(arg_df['arg-score-mace-p'].max(), arg_df['arg-score-mace-p'].min())

        # sentences_df = pd.read_json(data_path)
        # sentences_df['sent-class'] = sentences_df['sent-class'].map({'n': 0, 'c': 1, 'p': 2})
        # train_text, test_text, train_labels, test_labels = train_test_split(arg_df['arg-full'],
        #                                                                     arg_df['arg-score-mace-p'],
        #                                                                     random_state=2018,
        #                                                                     shuffle=True,
        #                                                                     test_size=test_size,
        #                                                                     )
        # train_text = arg_df[arg_df['train']]
        # test_text = arg_df[False == arg_df['train']]
        # train_labels = train_text['arg-score-mace-p']
        # test_labels = test_text['arg-score-mace-p']
        # train_text = train_text[['arg-text', 'arg-topic']]
        # test_text = test_text[['arg-text', 'arg-topic']]
        # print('Number of training rows : ', len(train_text))
        # print('Number of testing rows : ', len(test_text))
        #
        # self.train_text, self.train_labels = train_text, train_labels
        # self.test_text, self.test_labels = test_text, test_labels
        arg_df = arg_df[arg_df['train']]
        self.train_text = arg_df[['arg-text', 'arg-topic', 'arg-score-mace-p']]
        # self.TokenizeAndEncode()

    # def TokenizeAndEncode(self):
    #     if 'bert-base-uncased' == TRANSFORMERS_MODEL_NAME:
    #         tokenizer = BertTokenizerFast.from_pretrained(TRANSFORMERS_MODEL_NAME, do_lower_case=True)
    #
    #     elif 'distilbert-base-uncased' == TRANSFORMERS_MODEL_NAME:
    #         tokenizer = DistilBertTokenizerFast.from_pretrained(TRANSFORMERS_MODEL_NAME, do_lower_case=True)
    #
    #     elif 'roberta-base' == TRANSFORMERS_MODEL_NAME:
    #         tokenizer = RobertaTokenizerFast.from_pretrained(TRANSFORMERS_MODEL_NAME, do_lower_case=True)
    #
    #     elif 'distilroberta-base' == TRANSFORMERS_MODEL_NAME:
    #         tokenizer = RobertaTokenizerFast.from_pretrained(TRANSFORMERS_MODEL_NAME, do_lower_case=True)
    #
    #     # tokenize and encode sequences in the training se
    #     self.tokens_train = tokenizer.batch_encode_plus(
    #         self.train_text.tolist(),
    #         max_length=max_seq_len,
    #         add_special_tokens=True,
    #         padding=True,
    #         truncation=True,
    #         return_token_type_ids=False
    #     )
    #
    #     # tokenize and encode sequences in the validation set
    #     self.tokens_test = tokenizer.batch_encode_plus(
    #         self.test_text.tolist(),
    #         max_length=max_seq_len,
    #         add_special_tokens=True,
    #         padding=True,
    #         truncation=True,
    #         return_token_type_ids=False
    #     )

    def GetDataLoaders(self, batch_size=16):
        train_examples = []
        arguments = self.train_text['arg-text']
        topics = self.train_text['arg-topic']
        scores = self.train_text['arg-score-mace-p']
        for argument, topic, score in zip(arguments, topics, scores):
            train_examples.append(InputExample(texts=[argument, topic], label=0))

        train_dataloader = DataLoader(train_examples, batch_size=batch_size)

        # test_dataloader = DataLoader(test_data, batch_size=batch_size)

        return train_examples, train_dataloader

    # def GetClassWeights(self):
    #     # compute the class weights
    #     class_wts = compute_class_weight('balanced', np.unique(self.train_labels), self.train_labels)
    #     # convert class weights to tensor
    #     weights = torch.tensor(class_wts, dtype=torch.float)
    #     weights = weights.to(device)
    #     return weights


class RowSentencesHandler():
    def __init__(self):
        if 'bert-base-uncased' == TRANSFORMERS_MODEL_NAME:
            tokenizer = BertTokenizerFast.from_pretrained(TRANSFORMERS_MODEL_NAME, do_lower_case=True)

        elif 'distilbert-base-uncased' == TRANSFORMERS_MODEL_NAME:
            tokenizer = DistilBertTokenizerFast.from_pretrained(TRANSFORMERS_MODEL_NAME, do_lower_case=True)

        elif 'roberta-base' == TRANSFORMERS_MODEL_NAME:
            tokenizer = RobertaTokenizerFast.from_pretrained(TRANSFORMERS_MODEL_NAME, do_lower_case=True)

        elif 'distilroberta-base' == TRANSFORMERS_MODEL_NAME:
            tokenizer = RobertaTokenizerFast.from_pretrained(TRANSFORMERS_MODEL_NAME, do_lower_case=True)

        self.tokenizer = tokenizer

    def GetDataLoader(self, sentences, labels=None):
        tokens = self.tokenizer.batch_encode_plus(
            sentences,
            max_length=max_seq_len,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            return_token_type_ids=False
        )

        sequence = torch.tensor(tokens['input_ids'])  # .to(device)
        mask = torch.tensor(tokens['attention_mask'])  # .to(device)
        if labels:
            y_labels = torch.tensor(labels)

        # wrap tensors
        if labels:
            data = TensorDataset(sequence, mask, y_labels)
        else:
            data = TensorDataset(sequence, mask)

        # sampler for sampling the data during training
        sampler = SequentialSampler(data)
        # dataLoader for validation set
        dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
        return dataloader
