from src.transformers_ML.config import *
from src.transformers_ML.data_loader import DataLoadHandler
from src.transformers_ML.utils import *




models_dir = ARG_EXTRACTION_ROOT_DIR + '/models/'
output_dir = ARG_EXTRACTION_ROOT_DIR + '/results-output/'
news_output_dir = output_dir + 'stock-market-news'
NUM_LABELS = 2


if 'bert-base-uncased' == TRANSFORMERS_MODEL_NAME:
    model = BertForSequenceClassification.from_pretrained(TRANSFORMERS_MODEL_NAME, num_labels = NUM_LABELS,
                                                        output_attentions = False, output_hidden_states = False)
                                                        # attention_probs_dropout_prob = 0.1, hidden_dropout_prob = 0.1)

elif 'distilbert-base-uncased' == TRANSFORMERS_MODEL_NAME:
    model = DistilBertForSequenceClassification.from_pretrained(TRANSFORMERS_MODEL_NAME, num_labels = NUM_LABELS,
                                                        output_attentions = False, output_hidden_states = False)

elif 'roberta-base' == TRANSFORMERS_MODEL_NAME:
    model = RobertaForSequenceClassification.from_pretrained(TRANSFORMERS_MODEL_NAME, num_labels = NUM_LABELS,
                                                        output_attentions = False, output_hidden_states = False)

elif 'distilroberta-base' == TRANSFORMERS_MODEL_NAME:
    model = RobertaForSequenceClassification.from_pretrained(TRANSFORMERS_MODEL_NAME, num_labels = NUM_LABELS,
                                                        output_attentions = False, output_hidden_states = False)


model = model.to(device)

def train_model(epochs=3):
    data_loader = DataLoadHandler()
    train_dataloader, test_dataloader = data_loader.GetDataLoaders()
    # class_weights = data_loader.GetClassWeights()

    # define the optimizer lr=2e-5
    optimizer = AdamW(model.parameters(), lr = 5e-5, eps = 1e-8)
    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * epochs
    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)

    # Store the average loss after each epoch so we can plot them.
    loss_values = []
    for epoch_i in range(epochs):
        # ========================================
        #               Training
        # ========================================
        # Perform one full pass over the training set.
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')
        train_loss = TrainBertSeqCl(model, train_dataloader, optimizer, scheduler) #, class_weights)
        loss_values.append(train_loss)

        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.
        # print('validation with training data: ')
        # EvaluateBertSeqCl(model, train_dataloader)


        # ========================================
        #             Model Saving
        # ========================================
        model_path = '{}{}_essays_a_c_p_epoch_{}_weighted.pt'.format(models_dir, TRANSFORMERS_MODEL_NAME, epoch_i)
        torch.save(model.state_dict(), model_path)
    print("Training complete!!") 

    print('validation with testing data: ')
    EvaluateBertSeqCl(model, test_dataloader)


def train_model_double(epochs=3):
    data_loader = DataLoadHandler()
    train_dataloader, test_dataloader = data_loader.GetDataLoaders()
    # class_weights = data_loader.GetClassWeights()

    # define the optimizer lr=2e-5
    optimizer = AdamW(model.parameters(), lr = 5e-5, eps = 1e-8)
    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * epochs
    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)

    # Store the average loss after each epoch so we can plot them.
    loss_values = []
    for epoch_i in range(epochs):
        # ========================================
        #               Training
        # ========================================
        # Perform one full pass over the training set.
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')
        train_loss = TrainBertSeqCl(model, train_dataloader, optimizer, scheduler) #, class_weights)
        loss_values.append(train_loss)

        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.
        # print('validation with training data: ')
        # EvaluateBertSeqCl(model, train_dataloader)


        # ========================================
        #             Model Saving
        # ========================================
        model_path = '{}{}_essays_a_c_p_epoch_{}_weighted.pt'.format(models_dir, TRANSFORMERS_MODEL_NAME, epoch_i)
        torch.save(model.state_dict(), model_path)
    print("Training complete!!") 

    print('validation with testing data: ')
    EvaluateBertSeqCl(model, test_dataloader)

def train_model_double_stage2(epochs=3):
    print('Stage 2 from doubly model')
    
    model = DistilBertForSequenceClassification.from_pretrained('bert_model_double')
   

    data_loader = DataLoadHandler()
    train_dataloader, test_dataloader = data_loader.GetDataLoaders()
    # class_weights = data_loader.GetClassWeights()

    # define the optimizer lr=2e-5
    optimizer = AdamW(model.parameters(), lr = 5e-5, eps = 1e-8)
    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * epochs
    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)

    # Store the average loss after each epoch so we can plot them.
    loss_values = []
    for epoch_i in range(epochs):
        # ========================================
        #               Training
        # ========================================
        # Perform one full pass over the training set.
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')
        train_loss = TrainBertSeqCl(model, train_dataloader, optimizer, scheduler) #, class_weights)
        loss_values.append(train_loss)

        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.
        # print('validation with training data: ')
        # EvaluateBertSeqCl(model, train_dataloader)


        # ========================================
        #             Model Saving
        # ========================================
        model_path = '{}{}_essays_a_c_p_epoch_{}_weighted.pt'.format(models_dir, TRANSFORMERS_MODEL_NAME, epoch_i)
        torch.save(model.state_dict(), model_path)
    print("Training complete!!") 

    print('validation with testing data: ')
    EvaluateBertSeqCl(model, test_dataloader)

# for stacking
def bert_train_and_predict(train_sentences, y_train, test_sentences, epochs=2):
    model = DistilBertForSequenceClassification.from_pretrained(TRANSFORMERS_MODEL_NAME, num_labels = NUM_LABELS,
                                                        output_attentions = False, output_hidden_states = False)
    model = model.to(device)
    
    train_sents = [sent['sent-text'] for sent in train_sentences]
    test_sents = [sent['sent-text'] for sent in test_sentences]
    data_loader_handler = RowSentencesHandler()
    train_dataloader = data_loader_handler.GetDataLoader(train_sents, y_train)

    # define the optimizer lr=2e-5
    optimizer = AdamW(model.parameters(), lr = 5e-5, eps = 1e-8)
    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * epochs
    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)

    loss_values = []
    for epoch_i in range(epochs):
        # ========================================
        #               Training
        # ========================================
        # Perform one full pass over the training set.
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')
        train_loss = TrainBertSeqCl(model, train_dataloader, optimizer, scheduler) #, class_weights)
        loss_values.append(train_loss)

    print("Training complete!!") 

    y_preds = Predict(model, test_sents, logits_enable=True)
    return model, y_preds


# for first stage
def bert_test_first_stage(test_sentences):
    print('Stage one BERT testing')
    
    model = DistilBertForSequenceClassification.from_pretrained('bert_model_double')
    
    test_sents = [sent['sent-text'] for sent in test_sentences]
    #data_loader_handler = RowSentencesHandler()

    print("Loading complete!!") 

    y_preds = Predict(model, test_sents, logits_enable=True)
    y_result = [0 if pair[0] > pair[1] else 1 for pair in y_preds]
    return y_result
    
    
# for second stage
def bert_test_second_stage(test_sentences):
    print('Stage two BERT testing')
    
    model = DistilBertForSequenceClassification.from_pretrained('models\\bert_model_double_stage2')
    
    test_sents = [sent['sent-text'] for sent in test_sentences]
    #data_loader_handler = RowSentencesHandler()

    print("Loading complete!!") 

    y_preds = Predict(model, test_sents, logits_enable=True)
    y_result = [0 if pair[0] > pair[1] else 1 for pair in y_preds]
    #print(y_preds)
    return y_result
