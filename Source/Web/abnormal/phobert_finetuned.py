import torch.nn as nn

class PhoBERT_finetuned(nn.Module):
    def __init__(self, phobert, hidden_size, num_class):
        """
        Initializes the PhoBERT_finetuned model.

        Args:
            phobert (PhoBERT model): Pretrained PhoBERT model.
            hidden_size (int): Size of the hidden layer.
            num_class (int): Number of output classes.
        """
        super(PhoBERT_finetuned, self).__init__()
        self.phobert = phobert
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.layer1 = nn.Linear(768, hidden_size)  # Assuming PhoBERT output size is 768
        self.layer2 = nn.Linear(hidden_size, num_class)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        """
        Defines the forward pass of the model.

        Args:
            sent_id (tensor): Input tensor for sentence IDs.
            mask (tensor): Attention mask tensor.

        Returns:
            tensor: Log probabilities of class predictions.
        """
        _, cls_hs = self.phobert(sent_id, attention_mask=mask, return_dict=False)
        x = self.layer1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.softmax(x)
        return x
