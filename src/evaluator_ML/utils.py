from transformers import Trainer

from src.evaluator_ML.config import *
from src.evaluator_ML.data_loader import RowSentencesHandler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


def mapping(y):
    conversion = ['n', 'c', 'p']
    ans = [conversion[int(p)] for p in y]
    return ans


# Function to calculate the measure of our predictions vs labels

def compute_metrics_for_regression(eval_pred):
    logits, labels = eval_pred
    labels = labels.reshape(-1, 1)

    mse = mean_squared_error(labels, logits)
    mae = mean_absolute_error(labels, logits)
    r2 = r2_score(labels, logits)
    single_squared_errors = ((logits - labels).flatten() ** 2).tolist()

    # Compute accuracy
    # Based on the fact that the rounded score = true score only if |single_squared_errors| < 0.5
    accuracy = sum([1 for e in single_squared_errors if e < 0.25]) / len(single_squared_errors)

    return {"mse": mse, "mae": mae, "r2": r2, "accuracy": accuracy}


#########################################################
# Function to train the bert for sequence regression
# model
#########################################################
def TrainBertSeqReg(model, train_dataloader, optimizer, scheduler):  # , class_weights):
    # Measure how long the training epoch takes.
    t0 = datetime.datetime.now()

    # Reset the total loss for this epoch.
    total_loss = 0
    model.loss_fct = nn.MSELoss()
    model.train()

    # For each batch of training data...
    for step, batch in enumerate(train_dataloader):

        # Progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = datetime.datetime.now() - t0

            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {}.'.format(step, len(train_dataloader), elapsed))

        # Unpack this training batch from our dataloader.
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using the 
        # `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids
        #   [1]: attention masks
        #   [2]: labels
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        # Always clear any previously calculated gradients before performing a
        # backward pass. PyTorch doesn't do this automatically because 
        # accumulating the gradients is "convenient while training RNNs". 
        # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
        model.zero_grad()

        # Perform a forward pass (evaluate the model on this training batch).
        # This will return the loss (rather than the model output) because we
        # have provided the `labels`.
        # The documentation for this `model` function is here: 
        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
        outputs = model(b_input_ids,
                        # token_type_ids=None,
                        attention_mask=b_input_mask)
        # print('output')
        # print(outputs)
        # b_logits = outputs[0]
        # # The call to `model` always returns a tuple, so we need to pull the
        # # loss value out of the tuple.
        # loss_fun = nn.MSELoss()
        # loss = loss_fun(b_logits.squeeze(), b_labels.squeeze())

        b_logits = outputs[0]  # Shape: (batch_size, sequence_length, hidden_size)

        # Reshape the outputs to fit into the fully connected NN
        batch_size, sequence_length, hidden_size = b_logits.size()
        b_logits = b_logits.view(batch_size, sequence_length * hidden_size)

        # Pass the transformer outputs through the fully connected NN
        input_size = sequence_length * hidden_size
        hidden_size = 128  # Example hidden size
        output_size = 10  # Example output size
        fc_nn = FullyConnectedNN(input_size, hidden_size, output_size)
        nn_outputs = fc_nn(b_logits)

        # Print the output of the fully connected NN
        print(nn_outputs)

        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value 
        # from the tensor.
        total_loss += nn_outputs.item()

        # Perform a backward pass to calculate the gradients.
        nn_outputs.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer.step()

        # Update the learning rate.
        scheduler.step()

    # Calculate the average loss over the training data.
    avg_train_loss = total_loss / len(train_dataloader)

    # Store the loss value for plotting the learning curve.
    # loss_values.append(avg_train_loss)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(datetime.datetime.now() - t0))

    return avg_train_loss


#########################################################
# Function to evaluate the bert for sequence regression
# model
#########################################################
def EvaluateBertSeqReg(model, validation_dataloader):
    print("")
    print("Running Validation...")

    t0 = datetime.datetime.now()

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()

    # Tracking variables 
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    total_labels, total_preds = np.array([]), np.array([])
    # Evaluate data for one epoch
    for batch in validation_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            # This will return the logits rather than the loss because we have
            # not provided labels.
            # token_type_ids is the same as the "segment ids", which 
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            # The documentation for this `model` function is here: 
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            outputs = model(b_input_ids,
                            # token_type_ids=None, 
                            attention_mask=b_input_mask)

        # Get the "logits" output by the model. The "logits" are the output
        # values prior to applying an activation function like the softmax.
        logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        total_labels = np.append(total_labels, label_ids.flatten())
        total_preds = np.append(total_preds, np.argmax(logits, axis=1).flatten())

        # Calculate the loss for this batch of test sentences.
        # loss_fun = nn.MSELoss()
        # loss = loss_fun(logits.squeeze(), b_labels.squeeze())
        
        # Accumulate the total loss.
        # eval_loss += loss.item()

        # Track the number of batches
        nb_eval_steps += 1

    
    # Report the final accuracy for this validation run.
    # print("  Validation took: {:}".format(datetime.datetime.now() - t0))
    # print("  Accuracy: {0:.4f}".format(eval_accuracy / nb_eval_steps))
    # print("  Accuracy-2: {0:.4f}".format(flat_accuracy(total_preds, total_labels)))
    return total_preds, total_labels


#########################################################
# Function to use the model for prediction
#########################################################
def Predict(model, sentences, logits_enable=False):
    # model.eval()

    text_handler = RowSentencesHandler()
    dataloader = text_handler.GetDataLoader(sentences)
    prediction_result = np.array([])
    logits_result = np.array([[0]])

    for batch in dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        input_ids, input_mask = batch

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=input_mask)

        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        prediction_result = np.append(prediction_result, np.argmax(logits, axis=1).flatten())
        logits_result = np.append(logits_result, logits, axis=0)

    if logits_enable:
        return logits_result[1:].tolist()

    return prediction_result.tolist()


#########################################################
# Fully Connected NN Class
#########################################################
class FullyConnectedNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FullyConnectedNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

