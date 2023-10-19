import optuna
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from BaseModel import BinaryClassifier
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from param import *
from ConfigModel import *

# Define the device (CPU or GPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAINLOADER = DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True)
TESTLOADER = DataLoader(data_test, batch_size=BATCH_SIZE, shuffle=False)


def objective(trial):
    """Define the objective function"""

    model = BinaryClassifier().to(DEVICE)
    
    # Suggest hyperparameters for the chosen optimizer
    optimizer_type = trial.suggest_categorical('optimizer_type', ['Adam', 'SGD'])

    if optimizer_type == 'Adam':
        learning_rate = trial.suggest_float('adam_learning_rate', 1e-5, 1.0, log=True)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
    elif optimizer_type == 'SGD':
        learning_rate = trial.suggest_float('sgd_learning_rate', 1e-5, 1.0, log=True)
        momentum = trial.suggest_float('sgd_momentum', 0.0, 1.0)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    criterion = nn.CrossEntropyLoss().to(DEVICE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

    performance = []
    best_validation_loss = np.inf
    epochs_without_improvement = 0
    early_stopping_patience = 5
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        predictions_train = []
        true_labels_train = []

        for inputs, labels in TRAINLOADER:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE).long()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss_train = criterion(outputs, labels)
            loss_train.backward()
            optimizer.step()

            running_loss += loss_train.item()
            _, predicted = torch.max(outputs.data, 1)
            predictions_train.extend(predicted.tolist())
            true_labels_train.extend(labels.tolist())

        train_f1 = f1_score(true_labels_train, predictions_train, average='weighted')

        if scheduler is not None:
            scheduler.step()

        model.eval()
        total_loss_test = 0.0
        total_samples = 0
        predictions_test = []
        true_labels_test = []

        with torch.no_grad():
            for inputs, labels in TESTLOADER:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE).long()
                outputs = model(inputs)
                loss_test = criterion(outputs, labels)
                total_loss_test += loss_test.item() * len(labels)
                total_samples += len(labels)

                _, predicted = torch.max(outputs.data, 1)
                predictions_test.extend(predicted.tolist())
                true_labels_test.extend(labels.tolist())

        avg_loss_test = total_loss_test / total_samples

        test_f1 = f1_score(true_labels_test, predictions_test, average='weighted')

        performance.append([running_loss / len(TRAINLOADER), avg_loss_test, train_f1, test_f1])
        # print(f"Epoch [{epoch+1}/{EPOCHS}] Loss Train: {performance[epoch][0] :.4f} Loss Test: {performance[epoch][1]:.4f} F1 Train: {performance[epoch][2]:.4f} F1 Test: {performance[epoch][3] :.4f}")
            # Check if validation loss has improved
        if avg_loss_test < best_validation_loss:
            best_validation_loss = avg_loss_test
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # If no improvement for a certain number of epochs, stop training
        if epochs_without_improvement >= early_stopping_patience:
            # print(f"Early stopping after {early_stopping_patience} epochs without improvement.")
            break

    return train_f1



study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)


# Best value for hyperparameter

print("Best value for each hyperparameter:", study.best_params)
print("F1-score", study.best_value)