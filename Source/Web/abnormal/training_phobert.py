import torch
from transformers import AutoModel, AutoTokenizer, AdamW
from torch import nn as nn
import json
from torch.utils.data import DataLoader, TensorDataset
from .phobert_finetuned import PhoBERT_finetuned

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Load train and validation dataset
with open('content.json', 'r', encoding="utf-8") as c:
    contents = json.load(c)
with open('val_content.json', 'r', encoding="utf-8") as v:
    val_contents = json.load(v)
# Load model PhoBERT and its tokenizer
phobert = AutoModel.from_pretrained('vinai/phobert-base')
tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')

# Process the train dataset:
tags = []
X = []
y = []

for content in contents['intents']:
    tag = content['tag']
    for pattern in content['patterns']:
        X.append(pattern)
        tags.append(tag)

tags_set = sorted(set(tags))

for tag in tags:
    label = tags_set.index(tag)
    y.append(label)
token_train = {}
token_train = tokenizer.batch_encode_plus(
    X,
    max_length=13,
    padding='max_length',
    truncation=True
)
X_train_mask = torch.tensor(token_train['attention_mask'])
X_train = torch.tensor(token_train['input_ids'])
y_train = torch.tensor(y)

# Process the validation dataset:
tags_val = []
X_val = []
y_val = []

for val_content in val_contents['intents']:
    tag_val = val_content['tag']
    for val_pattern in val_content['patterns']:
        X_val.append(val_pattern)
        tags_val.append(tag_val)

for tag_val in tags_val:
    label = tags_set.index(tag_val)
    y_val.append(label)
token_val = {}
token_val = tokenizer.batch_encode_plus(
    X_val,
    max_length=13,
    padding='max_length',
    truncation=True
)
X_val_mask = torch.tensor(token_val['attention_mask'])
X_val = torch.tensor(token_val['input_ids'])
y_val = torch.tensor(y_val)

# Hyperparameter:
batch_size = 8
hidden_size = 512
num_class = len(tags_set)
lr = 7.5e-5
num_epoch = 500

dataset = TensorDataset(X_train, X_train_mask, y_train)
train_data = DataLoader(dataset=dataset, batch_size=batch_size,
                        shuffle=True)
val_dataset = TensorDataset(X_val, X_val_mask, y_val)
val_data = DataLoader(dataset=val_dataset, batch_size=batch_size,
                      shuffle=True)

# Model:
for param in phobert.parameters():
    param.requires_grad = False
model = PhoBERT_finetuned(phobert, hidden_size=hidden_size,
                          num_class=num_class)
model = model.to(device)
optimizer = AdamW(model.parameters(), lr=lr)
loss_f = nn.NLLLoss()


def train():
    print("Training...")
    model.train()
    total_loss = 0
    for (step, batch) in enumerate(train_data):
        # push the batch to gpu
        batch = [r.to(device) for r in batch]
        sent_id, mask, labels = batch
        # clear previously calculated gradients
        model.zero_grad()
        pred = model(sent_id, mask)
        loss = loss_f(pred, labels)
        total_loss += loss.item()
        loss.backward()
        # clip the gradients to 1.0.
        # It helps in preventing the exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    avg_loss = total_loss/len(train_data)
    return avg_loss


def evaluate():
    print("Evaluating...")
    # deactivate dropout layers
    model.eval()
    total_loss = 0
    # iterate over batches
    for (step, batch) in enumerate(val_data):
        # push the batch to gpu
        batch = [t.to(device) for t in batch]
        sent_id_val, mask_val, labels_val = batch

        # deactivate autograd
        with torch.no_grad():
            # model predictions
            preds = model(sent_id_val, mask_val)
            # compute the validation loss between actual and predicted values
            loss = loss_f(preds, labels_val)
            total_loss = total_loss + loss.item()
            preds = preds.detach().cpu().numpy()

    # compute the validation loss of the epoch
    avg_loss = total_loss / len(val_data)
    return avg_loss

def StartTrain():
    best_valid_loss = float('inf')
    train_losses = []
    valid_losses = []

    for epoch in range(num_epoch):
        print('\n Epoch {:}/{:}'.format(epoch + 1, num_epoch))
        train_loss = train()
        # evaluate model
        valid_loss = evaluate()
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'saved_weights.pth')
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        print(f'\nTraining Loss: {train_loss:.3f}')
        print(f'Validation Loss: {valid_loss:.3f}')

# StartTrain()