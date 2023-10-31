from src.data_visualizer.curve_visualizer import TrainingValidationLoss, ValidationLoss, PearsonMeasure
from src.evaluator_ML.data_loader import DataLoadHandler
from src.evaluator_ML.utils import *

models_dir = ARG_EXTRACTION_ROOT_DIR + '/models/'
output_dir = ARG_EXTRACTION_ROOT_DIR + '/results-output/'
news_output_dir = output_dir + 'stock-market-news'
NUM_LABELS = 1

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
config = model.config
# Modify the activation attribute
config.activation = "sigmoid"
# Create a new model with the modified configuration
model = BertForSequenceClassification(config)
model = model.to(device)


def train_model(epochs=5):
    data_loader = DataLoadHandler()
    train_dataloader, test_dataloader = data_loader.GetDataLoaders()

    # class_weights = data_loader.GetClassWeights()

    # define the optimizer lr=2e-5
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * epochs
    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)

    # Store the average loss after each epoch so we can plot them.
    train_loss_values = []
    val_loss_values = []
    for epoch_i in range(epochs):
        # ========================================
        #               Training
        # ========================================
        # Perform one full pass over the training set.
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')
        train_loss = TrainBertSeqReg(model, train_dataloader, optimizer, scheduler)  # , class_weights)
        train_loss_values.append(train_loss)

        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.
        print('validation with training data: ')
        val_loss = EvaluateBertSeqReg(model, test_dataloader)
        print('Testing loss : {}', val_loss)

        val_loss_values.append(val_loss)

        # ========================================
        #             Model Saving
        # ========================================
        model_path = '{}{}_ibm_scorer_30k_epoch_{}_weighted.pt'.format(models_dir, TRANSFORMERS_MODEL_NAME, epoch_i)
        torch.save(model.state_dict(), model_path)
    print("Training complete!!")
    TrainingValidationLoss(train_loss_values, val_loss_values)
    print('validation with testing data: ')
    print('Testing Loss Value : ', EvaluateBertSeqReg(model, test_dataloader))


def train_existed_model(epochs=5):
    data_loader = DataLoadHandler()
    train_dataloader, test_dataloader = data_loader.GetDataLoaders()

    # class_weights = data_loader.GetClassWeights()

    # define the optimizer lr=2e-5
    # optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    # # Total number of training steps is number of batches * number of epochs.
    # total_steps = len(train_dataloader) * epochs
    # # Create the learning rate scheduler.
    # scheduler = get_linear_schedule_with_warmup(optimizer,
    #                                             num_warmup_steps=0,  # Default value in run_glue.py
    #                                             num_training_steps=total_steps)

    # Store the average loss after each epoch so we can plot them.
    act_values = []
    val_values = []
    for epoch_i in range(epochs):
        # ========================================
        #               Loading the model
        # ========================================
        # Perform one full pass over the training set.
        model_path = '{}{}_ibm_scorer_30k_epoch_{}_weighted.pt'.format(models_dir, TRANSFORMERS_MODEL_NAME, epoch_i)

        model.load_state_dict(torch.load(model_path))
        model.eval()
        print('model {} loaded successfully!'.format(epoch_i))
        # ========================================
        #               Training
        # ========================================
        # Perform one full pass over the training set.
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        # print('Training...')
        # train_loss = TrainBertSeqReg(model, train_dataloader, optimizer, scheduler)  # , class_weights)
        # train_loss_values.append(train_loss)

        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.
        print('validation with testing data: ')
        # train_loss_values.append(EvaluateBertSeqReg(model, train_dataloader))
        pred, act = EvaluateBertSeqReg(model, test_dataloader)
        for p, a in zip(pred, act):
            print(p, a)
        act_values.append(act)
        val_values.append(pred)
        # ========================================
        #             Model Saving
        # ========================================
        # model_path = '{}{}_essays_a_c_p_epoch_{}_weighted.pt'.format(models_dir, TRANSFORMERS_MODEL_NAME, epoch_i)
        # torch.save(model.state_dict(), model_path)
    print("Testing complete!!")
    # ValidationLoss(val_loss_values)
    PearsonMeasure(val_values, act_values)
    # SpearmanMeasure(val_loss_values)
    for val in val_values:
        print(val)
    # print('validation with testing data: ')
    # print('Testing Loss Value : ', EvaluateBertSeqReg(model, test_dataloader))


# for handle qureies
def prediction_handler(test_sentences):
    print('Stage one BERT testing')

    model.load_state_dict(torch.load(models_dir + 'bert-base-uncased_ibm_scorer_30k_epoch_2_weighted.pt'))

    # test_sents = [sent['sent-text'] for sent in test_sentences]
    # data_loader_handler = RowSentencesHandler()

    print("Loading complete!!")

    y_preds = Predict(model, test_sentences, logits_enable=True)
    print('Max prediction: ', max(y_preds), 'Min prediction: ', min(y_preds))
    return y_preds