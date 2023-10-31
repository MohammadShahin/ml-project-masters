import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
import transformers

from transformers import AutoModel, BertTokenizerFast, BertTokenizer, BertForSequenceClassification, BertConfig
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast, RobertaForSequenceClassification, RobertaTokenizerFast

from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import en_core_web_sm

import time
import datetime
import random
import os
import sys
import json
import pickle
import pathlib

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




