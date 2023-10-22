from transformers import RobertaForSequenceClassification,AutoModelForSequenceClassification
from transformers import AdamW
from transformers import get_scheduler,AutoTokenizer
import torch
from datasets import load_metric
from .Remove import remove_punctuation
from tqdm.auto import tqdm

checkpoint = "wonrax/phobert-base-vietnamese-sentiment"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
class Model:
    def __init__(self,name_model="wonrax/phobert-base-vietnamese-sentiment"):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print(self.device)
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(name_model)
            print('Load Model: ',name_model)
        except:
            print('Load Model: ',name_model)
            self.model = RobertaForSequenceClassification.from_pretrained(name_model)

        self.model.to(self.device)

    def Train(self,train_dataloader,learning_rate = 5e-5,epoch = 3,save_model = ''):
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        num_epochs = epoch
        num_training_steps = num_epochs * len(train_dataloader)
        lr_scheduler = get_scheduler(
            "polynomial",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps
        )

        progress_bar = tqdm(range(num_training_steps))
        self.model.train()
        print('Training on ',epoch,' epoch')
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in train_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
            average_loss = total_loss / len(train_dataloader)
            print(f"Epoch {epoch + 1} - Average Loss: {average_loss:.4f}")

        if save_model != '':
            print('Saved ',save_model)
            self.model.save_pretrained(save_model)



    def Train_CrossValidation(self, train_dataloader, learning_rate=5e-5, epoch=3, save_model=''):
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        num_epochs = epoch

        # Split the data into k folds for cross-validation
        k = 5  # You can choose the number of folds as desired
        fold_size = len(train_dataloader) // k

        for fold in range(k):
            # Separate the current fold for validation
            valid_start = fold * fold_size
            valid_end = (fold + 1) * fold_size
            train_fold = train_dataloader[:valid_start] + train_dataloader[valid_end:]
            valid_fold = train_dataloader[valid_start:valid_end]

            num_training_steps = num_epochs * len(train_fold)
            lr_scheduler = get_scheduler(
                "linear",
                optimizer=optimizer,
                num_warmup_steps=0,
                num_training_steps=num_training_steps
            )

            progress_bar = tqdm(range(num_training_steps))
            self.model.train()
            print(f'Training on fold {fold + 1}')
            for epoch in range(num_epochs):
                total_loss = 0
                for batch in train_fold:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    total_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                average_loss = total_loss / len(train_fold)
                print(f"Fold {fold + 1} - Epoch {epoch + 1} - Average Loss: {average_loss:.4f}")

                # Perform validation on the current fold
                self.model.eval()
                with torch.no_grad():
                    total_valid_loss = 0
                    for batch in valid_fold:
                        batch = {k: v.to(self.device) for k, v in batch.items()}
                        outputs = self.model(**batch)
                        loss = outputs.loss
                        total_valid_loss += loss.item()
                    average_valid_loss = total_valid_loss / len(valid_fold)
                    print(f"Validation Loss for Fold {fold + 1} - Epoch {epoch + 1}: {average_valid_loss:.4f}")

        if save_model != '':
            print('Saved', save_model)
            self.model.save_pretrained(save_model)

    def Predict(self,input_text=''):
        if input_text == "":
            return '', ''
        inputs = tokenizer(input_text, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits.to("cpu")
        probabilities = torch.softmax(logits, dim=1).numpy()

        predicted_class = torch.argmax(logits, dim=1).item()

        label_names = ["Negative", "Neutral", "Positive"]
        predicted_label = label_names[predicted_class]

        return predicted_label, probabilities[0][predicted_class]

    def Eval(self,eval_dataloader):
        self.model.to(self.device)
        print(self.device)
        metric = load_metric("accuracy")
        self.model.eval()
        for batch in eval_dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])
        print(metric.compute())