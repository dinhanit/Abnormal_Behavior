
import torch
LEARNING_RATE= 0.001
EPOCHS=100
BATCH_SIZE=64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")