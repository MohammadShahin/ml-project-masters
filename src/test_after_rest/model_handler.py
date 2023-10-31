import torch
from sentence_transformers import SentenceTransformer, InputExample, LoggingHandler, losses, models, util
from torch.utils.data import DataLoader
from sentence_transformers.evaluation import TripletEvaluator, EmbeddingSimilarityEvaluator
from datetime import datetime
from zipfile import ZipFile

from sentence_transformers.datasets import SentenceLabelDataset
from sentence_transformers.datasets import NoDuplicatesDataLoader

import csv
import logging
import os
import sys

from src.test_after_rest.config import TRANSFORMERS_MODEL_NAME, ARG_EXTRACTION_ROOT_DIR, device

from src.test_after_rest.data_loader import DataLoadHandler

models_dir = ARG_EXTRACTION_ROOT_DIR + '/models/'
output_dir = ARG_EXTRACTION_ROOT_DIR + '/results-output/'
news_output_dir = output_dir + 'stock-market-news'
NUM_LABELS = 2

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)


def train_model(num_epochs=3, max_seq_length=256):
    ### Configure sentence transformers for training and train on the provided dataset
    # Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
    output_path = models_dir + TRANSFORMERS_MODEL_NAME + "-" + datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S")

    word_embedding_model = models.Transformer(TRANSFORMERS_MODEL_NAME)
    word_embedding_model.max_seq_length = max_seq_length

    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True,
                                   pooling_mode_cls_token=False,
                                   pooling_mode_max_tokens=False)

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    model.to(device)
    logger.info("Read train dataset")
    dataloader = DataLoadHandler()
    train_examples, train_dataloader = dataloader.GetDataLoaders()
    # for row in train_dataloader:
    #     print(row)
    #     train_examples.append(InputExample(texts=[row['arg-text'], row['arg-topic']], label=row['arg-score-mace-p']))

    regression_loss = losses.MSELoss(model=model)

    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(train_examples, name='regression-evaluator')
    warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data

    # Train the model
    model.fit(train_objectives=[(train_dataloader, regression_loss)],
              evaluator=evaluator,
              epochs=num_epochs,
              evaluation_steps=500,
              warmup_steps=warmup_steps,
              output_path=output_path)

# if __name__ == '__main__':
#     train_model()
