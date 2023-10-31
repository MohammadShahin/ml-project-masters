from src.data_visualizer.curve_visualizer import TrainingValidationLoss
from src.stance_ML.data_loader import DataLoadHandler
from src.stance_ML.utils import *

models_dir = ARG_EXTRACTION_ROOT_DIR + '/models/'
output_dir = ARG_EXTRACTION_ROOT_DIR + '/results-output/'
news_output_dir = output_dir + 'stock-market-news'
NUM_LABELS = 2

if 'bert-base-uncased' == TRANSFORMERS_MODEL_NAME:
    model = BertForSequenceClassification.from_pretrained(TRANSFORMERS_MODEL_NAME, num_labels=NUM_LABELS,
                                                          output_attentions=False, output_hidden_states=False)
    # attention_probs_dropout_prob = 0.1, hidden_dropout_prob = 0.1)

elif 'distilbert-base-uncased' == TRANSFORMERS_MODEL_NAME:
    model = DistilBertForSequenceClassification.from_pretrained(TRANSFORMERS_MODEL_NAME, num_labels=NUM_LABELS,
                                                                output_attentions=False, output_hidden_states=False)

elif 'roberta-base' == TRANSFORMERS_MODEL_NAME:
    model = RobertaForSequenceClassification.from_pretrained(TRANSFORMERS_MODEL_NAME, num_labels=NUM_LABELS,
                                                             output_attentions=False, output_hidden_states=False)

elif 'distilroberta-base' == TRANSFORMERS_MODEL_NAME:
    model = RobertaForSequenceClassification.from_pretrained(TRANSFORMERS_MODEL_NAME, num_labels=NUM_LABELS,
                                                             output_attentions=False, output_hidden_states=False)

model = model.to(device)


def train_model(epochs=5):
    data_loader = DataLoadHandler()
    train_dataloader, test_dataloader = data_loader.GetDataLoaders()
    # class_weights = data_loader.GetClassWeights()

    # define the optimizer lr=2e-5
    optimizer = AdamW(model.parameters(), lr=5e-5, eps=1e-8)
    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * epochs
    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)

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
        train_loss = TrainBertSeqCl(model, train_dataloader, optimizer, scheduler)  # , class_weights)
        loss_values.append(train_loss)

        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.
        print('validation with training data: ')
        EvaluateBertSeqCl(model, test_dataloader)

        # ========================================
        #             Model Saving
        # ========================================
        model_path = '{}{}_ibm_rank_stance_epoch_{}_weighted.pt'.format(models_dir, TRANSFORMERS_MODEL_NAME, epoch_i)
        torch.save(model.state_dict(), model_path)
        print("Training complete!!")
        # TrainingValidationLoss(loss_values, val_loss_values)
        print('validation with testing data: ')
        # print('Testing Loss Value : ', EvaluateBertSeqReg(model, test_dataloader))


# for first stage
def predict_stance(test_sentences, topic):
    print('Start Stance Prediction')

    model = DistilBertForSequenceClassification.from_pretrained('models\\model_stance')

    test_sents = [sent['sent-text'] + ' [SEP] ' + topic for sent in test_sentences]
    # data_loader_handler = RowSentencesHandler()

    print("Loading complete!!")

    y_preds = Predict(model, test_sents, logits_enable=True)
    y_result = [+1 if pair[0] > pair[1] else -1 for pair in y_preds]
    # print(y_preds)
    return y_result  # , y_preds
