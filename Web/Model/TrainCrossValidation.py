from ConfigModel import *
from sklearn.metrics import f1_score

from sklearn.model_selection import StratifiedKFold

# Define the number of folds

skf = StratifiedKFold(n_splits=n_folds, shuffle=True)

# Initialize lists to store evaluation results
fold_accuracies = []

for fold, (train_indices, val_indices) in enumerate(skf.split(data_train.features, data_train.labels)):
    # Create data loaders for the current fold
    train_subset = torch.utils.data.Subset(data_train, train_indices)
    val_subset = torch.utils.data.Subset(data_train, val_indices)
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Create a new model for each fold (if needed)
    model = BinaryClassifier()
    model.to(DEVICE)
    
    # Training loop for the current fold
    for epoch in range(EPOCHS):
        model.train()
        for batch in train_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Validation loop for the current fold
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    fold_accuracy = 100 * correct / total
    fold_accuracies.append(fold_accuracy)

    print(f'Fold {fold + 1} - Accuracy: {fold_accuracy:.2f}%')

# Calculate and print the mean accuracy across all folds
mean_accuracy = np.mean(fold_accuracies)
print(f'Mean Accuracy: {mean_accuracy:.2f}%')

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
print('F1 : ',test_f1)