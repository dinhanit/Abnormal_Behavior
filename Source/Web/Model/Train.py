from ConfigModel import *
from sklearn.metrics import f1_score
import pandas as pd
import argparse

performance= []

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



parser = argparse.ArgumentParser(description='Your script description')
parser.add_argument('--save-csv', type=str, default='', help='Path to save the performance CSV file')
parser.add_argument('--save-model', type=str, default='', help='Path to save the trained model')
args = parser.parse_args()

def Save_Perform():
    global performance
    df = pd.DataFrame(performance, columns=['Loss Train', 'Loss Test', 'F1 Train','F1 Test'])
    df.to_csv('Performance.csv',index=False)

if args.save_csv != "":
    Save_Perform()
if args.save_model != "":
    torch.save(model,'.model/weight')