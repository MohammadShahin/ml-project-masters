# import datetime
# import os
#
# import torch
# from torch.utils.data import Dataset, DataLoader
# from transformers import BertModel, BertTokenizer, AdamW
# import pandas as pd
#
# from src.evaluator_ML.data_loader import DataLoadHandler
#
# ARG_EXTRACTION_ROOT_DIR = os.path.abspath(os.getcwd())
# data_dir = ARG_EXTRACTION_ROOT_DIR + '/corpora/parsed-corpora/'
#
# #
# # class ArgumentDataset(Dataset):
# #     def __init__(self, sentences, scores, tokenizer, max_length):
# #         self.sentences = sentences
# #         self.scores = scores
# #         self.tokenizer = tokenizer
# #         self.max_length = max_length
# #
# #     def __len__(self):
# #         return len(self.sentences)
# #
# #     def __getitem__(self, idx):
# #         sentence = str(self.sentences[idx])
# #         score = float(self.scores[idx])
# #
# #         encoding = self.tokenizer.encode_plus(
# #             sentence,
# #             add_special_tokens=True,
# #             max_length=self.max_length,
# #             padding='max_length',
# #             truncation=True,
# #             return_tensors='pt'
# #         )
# #
# #         input_ids = encoding['input_ids'].squeeze()
# #         attention_mask = encoding['attention_mask'].squeeze()
# #
# #         return {
# #             'input_ids': input_ids,
# #             'attention_mask': attention_mask,
# #             'score': torch.tensor(score)
# #         }
#
#
# data_loader = DataLoadHandler()
# train_loader, val_loader = data_loader.GetDataLoaders()
# # Load the pre-trained BERT model
# model = BertModel.from_pretrained('bert-base-uncased')
#
# # Define the optimizer
# optimizer = AdamW(model.parameters(), lr=2e-5)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)
#
# num_epochs = 5
# print('Training ...')
# for epoch in range(num_epochs):
#     print(f'======Epoch {epoch + 1} / {num_epochs}=======')
#     model.train()
#     train_loss = 0.0
#     i = 1
#     t0 = datetime.datetime.now()
#
#     for step, batch in enumerate(train_loader):
#
#         # Progress update every 40 batches.
#         if step % 40 == 0 and not step == 0:
#             # Calculate elapsed time in minutes.
#             elapsed = datetime.datetime.now() - t0
#             # Report progress.
#             print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {}.'.format(step, len(train_loader), elapsed))
#
#         input_ids = batch[0].to(device)
#         attention_mask = batch[1].to(device)
#         scores = batch[2].to(device)
#
#         optimizer.zero_grad()
#
#         outputs = model(input_ids, attention_mask=attention_mask)
#         predictions = outputs.last_hidden_state.mean(dim=0)  # Average pooling over the sequence length
#         for x in predictions:
#             print(x[0].mean(dim=0))
#         loss = torch.nn.MSELoss()(predictions.squeeze(), scores.squeeze())
#
#         loss.backward()
#         optimizer.step()
#         print(f'batch {i} : {loss.item()}')
#         train_loss += loss.item()
#         i = i + 1
#     train_loss /= len(train_loader)
#
#     # Validation loop
#     print('Validation ...')
#     model.eval()
#     val_loss = 0.0
#     i = 1
#     with torch.no_grad():
#         for batch in val_loader:
#             input_ids = batch[0].to(device)
#             attention_mask = batch[1].to(device)
#             scores = batch[2].to(device)
#
#             outputs = model(input_ids, attention_mask=attention_mask)
#             outputs = outputs[0]
#             # predictions = outputs.last_hidden_state.mean(dim=1)
#             predictions = outputs
#             # print(predictions)
#             loss_fun = torch.nn.MSELoss()
#             # loss = loss_fun(predictions.squeeze(), scores.squeeze())
#             print(f'batch {i} : {loss.item()}')
#             val_loss += 0
#             i = i + 1
#
#         val_loss /= len(val_loader)
#
#     print(f'Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}')
# from back_end.run import app
# from model_manager.annotator import GetAnnotations
# from src.data_parser.utils import ProcessSingleEssay
#
#
# from back_end.run import run_server
# from model_manager.annotator import GetAnnotations, BuildTree
# from src.data_parser.utils import ProcessSingleEssay
#
#
# def tester():
#     file_path = "example_essay.txt"
#     file = open(file_path, "r")
#
#     # Step 2: Read the contents of the file
#     file_content = file.read()
#     essays_sentences, topic = ProcessSingleEssay(file_content)
#     for sent in essays_sentences:
#         print(sent['sent-text'])
#     # print(essays_sentences)
#     annotated = GetAnnotations(essays_sentences)
#     scored = BuildTree(annotated, topic)
#     for arg in scored:
#         if arg['annotation'] == 'PREMISE':
#             print(arg['score'])
#     # print(res)
#
from src.test_after_rest.model_handler import train_model

if __name__ == '__main__':
    train_model()
    # app.run(debug=True)
    # tester()
    # run_server()
    # arguments = [
    #     {
    #         'sent-text': 'it should be clear that practical skills foster problem-solving abilities which are '
    #                      'essential for professionals. ',
    #         'stance': 1
    #     },
    #     {
    #         'sent-text': 'playing football is better in driving cars ',
    #         'stance': 1
    #     },
    #     {
    #         'sent-text': 'superman has the ability to be software man, but he cannot play football anymore',
    #         'stance': 1
    #     }
    # ]
    # topic = 'Software Engineering students should learn practical skills more than academic ones in the Bachelor level'
    # # Load the model architecture
    #
    # res = calc_similarity(arguments, topic)
    # for arg, r in zip(arguments, res):
    #     print(arg['sent-text'])
    #     print(r)
    #     print()
# #
# from src.data_visualizer.ibm_visualizer import show_all_ibm
# from src.data_visualizer.ukp_visualizer import show_all_essays
# from src.data_visualizer.web_visualizer import show_all_web
# from src.evaluator_ML.model_handler import train_existed_model
#
# if __name__ == '__main__':
#     # app.run(debug=True)
#     # show_all_ibm()
#     train_existed_model()
