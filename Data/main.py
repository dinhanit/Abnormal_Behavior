import numpy as np 

def load_data(data):  #Fucntion to read "train_data.npy" and "test_data.npy"
    loaded_array = np.load(data, allow_pickle=True)
    X = loaded_array[0]
    y = loaded_array[1]
    
    # print(X.shape)
    # print(y.shape)
    return X, y 

#load_data('train_data.npy')