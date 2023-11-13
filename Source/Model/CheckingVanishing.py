"""
Module: checking_vanishing.py
Description: This module checks for vanishing gradients during training.
"""
from ConfigModel import *
from sklearn.metrics import f1_score
import pandas as pd
import argparse



num_params = sum(p.numel() for p in model.parameters())
performance = []
best_validation_loss = np.inf
epochs_without_improvement = 0
early_stopping_patience = 20
layer_gradient_norms = {name: [] for name, _ in model.named_parameters()}

epoch_param_gradient_norms = {name: [] for name, _ in model.named_parameters()}

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
    
    performance.append([running_loss / len(TRAINLOADER),avg_loss_test,train_f1,test_f1])
    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss Train: {performance[epoch][0] :.4f} Loss Test: {performance[epoch][1]:.4f} F1 Train: {performance[epoch][2]:.4f} F1 Test: {performance[epoch][3] :.4f}")

    # Calculate and save gradient norms at the end of each epoch
    for name, param in model.named_parameters():
        if param.grad is not None:
            gradient_norm = param.grad.norm().item()
            epoch_param_gradient_norms[name].append(gradient_norm)

    if avg_loss_test < best_validation_loss:
        best_validation_loss = avg_loss_test
        epochs_without_improvement = 0
        best_epoch = epoch
    else:
        epochs_without_improvement += 1
    if epochs_without_improvement >= early_stopping_patience:
        break


parser = argparse.ArgumentParser(description='Your script description')
parser.add_argument('--save-csv', type=str, default='', help='Path to save the performance CSV file')
parser.add_argument('--save-model', type=str, default='', help='Path to save the trained model')
args = parser.parse_args()

def Save_Gradient_Norms():
    df = pd.DataFrame(epoch_param_gradient_norms)
    df.to_csv('EpochGradientNorms.csv', index=False)


def Save_Perform():
    global performance
    df = pd.DataFrame(performance, columns=['Loss Train', 'Loss Test', 'F1 Train','F1 Test'])
    df.to_csv('Performance.csv',index=False)

if args.save_csv != "":
    Save_Perform()
    Save_Gradient_Norms()
if args.save_model != "":
    torch.save(model,'.model/weight')
print(f"Epoch with the best test loss: {best_epoch + 1} (Loss: {best_validation_loss:.4f})")