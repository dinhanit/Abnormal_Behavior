# from transformers import RobertaForSequenceClassification, AutoTokenizer
import torch
from .set_model import Model
from .vni_acronym import Acronym

# model = RobertaForSequenceClassification.from_pretrained("wonrax/phobert-base-vietnamese-sentiment")
# tokenizer = AutoTokenizer.from_pretrained("wonrax/phobert-base-vietnamese-sentiment", use_fast=False)

sentiment_model = Model('fine_tuned_model_best')
A = Acronym()
