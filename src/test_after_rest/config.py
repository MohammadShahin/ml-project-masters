import os
import random

import en_core_web_sm
import numpy as np
import torch

TRANSFORMERS_MODEL_NAME = 'distilbert-base-uncased'

# pathlib.Path(__file__).parent.absolute()
ARG_EXTRACTION_ROOT_DIR = os.path.abspath(os.getcwd())


# specify GPU
# device = torch.device("cuda")
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# device = torch.device("cpu")

nlp = en_core_web_sm.load()

# Set the seed value all over the place to make this reproducible.
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)












'''
model_type = 'bert'
model_name = PRETRAINED_BERT_MODEL
train_args = {
    'output_dir': f'{model_type}-{model_name}-outputs',

    'max_seq_length': 256,
    'num_train_epochs': 5,
    'train_batch_size': 16,
    'eval_batch_size': 32,
    'gradient_accumulation_steps': 1,
    'learning_rate': 5e-5,
    'save_steps': 50000,

    'wandb_project': 'ag-news-transformers-comparison',
    'wandb_kwargs': {'name': f'{model_type}-{model_name}'},
    'evaluate_during_training': True,
    'evaluate_during_training_steps': 1000,
    'reprocess_input_data': True,
    "save_model_every_epoch": False,
    'overwrite_output_dir': True,
    'no_cache': True,

    'use_early_stopping': True,
    'early_stopping_patience': 3,
    'manual_seed': 4,
}
'''