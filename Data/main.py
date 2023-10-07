import pickle

def load_data(data): #FUNCTION TO LOAD DATA 
    with open(data, "rb") as file:
    # Load the serialized object from the file
        loaded_data = pickle.load(file)
    #print(loaded_data)

    print("Loaded Done")
    return loaded_data

# load_data('train_data.pkl')
# load_data('test_data.pkl')
