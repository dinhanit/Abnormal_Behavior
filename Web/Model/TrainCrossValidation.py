from ConfigModel import *
from sklearn.metrics import f1_score
import argparse
import pandas as pd
from sklearn.model_selection import KFold
from param import *

num_folds = 5 

kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
fold = 0

for train_indices, val_indices in kf.split(x):
    train_features, train_labels = x[train_indices], y[train_indices]
    val_features, val_labels = x[val_indices], y[val_indices]

    train_dataset = CustomDataset(train_features, train_labels)
    val_dataset = CustomDataset(val_features, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model.to(DEVICE)

    fold += 1

    for epoch in range(EPOCHS):
        model.train()
        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to(DEVICE), batch_labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

        print(f'Fold [{fold}/{num_folds}] - Epoch [{epoch+1}/{EPOCHS}] - Training Loss: {loss.item():.4f}')

        # Evaluation on the validation set
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for val_batch_features, val_batch_labels in val_loader:
                val_batch_features, val_batch_labels = val_batch_features.to(DEVICE), val_batch_labels.to(DEVICE)
                val_outputs = model(val_batch_features)
                val_loss += criterion(val_outputs, val_batch_labels).item()
            val_loss /= len(val_loader)

        print(f'Fold [{fold}/{num_folds}] - Epoch [{epoch+1}/{EPOCHS}] - Validation Loss: {val_loss:.4f}')


parser = argparse.ArgumentParser(description='Your script description')
parser.add_argument('--save-csv', type=str, default='', help='Path to save the performance CSV file')
parser.add_argument('--save-model', type=str, default='', help='Path to save the trained model')
args = parser.parse_args()

# def Save_Perform():
#     global performance
#     df = pd.DataFrame(performance, columns=['Loss Train', 'Loss Validation', 'F1 Train', 'F1 Validation'])
#     df.to_csv(f'Performance_Fold{fold}.csv', index=False)
# if args.save_csv != "":
#     Save_Perform()
if args.save_model != "":
    torch.save(model,'weight')