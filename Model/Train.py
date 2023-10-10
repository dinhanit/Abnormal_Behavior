from ConfigModel import *

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    for inputs, labels in TRAINLOADER:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE).long()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss_train = criterion(outputs,labels)
        loss_train.backward()
        optimizer.step()
    if scheduler is not None:
        scheduler.step()
    
    model.eval() 
    with torch.no_grad():
        for data in TESTLOADER:
            inputs, labels = data
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE).long()
            outputs = model(inputs)
            loss_test = criterion(outputs, labels)
            
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss Train: {loss_train.item():.4f} Loss Test: {loss_test.item():.4f}")
