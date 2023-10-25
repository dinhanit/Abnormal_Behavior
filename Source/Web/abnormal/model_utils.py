from transformers import RobertaForSequenceClassification, AutoTokenizer
import torch
from .SetModel import Model
from .VniAcronym import Acronym

model = RobertaForSequenceClassification.from_pretrained("wonrax/phobert-base-vietnamese-sentiment")
tokenizer = AutoTokenizer.from_pretrained("wonrax/phobert-base-vietnamese-sentiment", use_fast=False)

model1 = Model('fine_tuned_model_best')
A = Acronym()


from .param import DEVICE
# model1 = torch.load("abmodel/weight").to(DEVICE)
