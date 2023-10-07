import torch

def load_data(data): 
    loaded_array = torch.load(data)
    X = loaded_array[0]
    y = loaded_array[1]
    
    # print(X.shape)
    # print(y.shape)
    return X, y 

# load_data('train_data.pth')
# load_data('test_data.pth')