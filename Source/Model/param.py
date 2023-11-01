
import torch
LEARNING_RATE= 0.0001
EPOCHS=100
BATCH_SIZE=32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")