import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import DataLoader, random_split, Dataset
import os, json
import logging
from datetime import datetime

handler = logging.FileHandler('train.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

class GPT2Dataset(Dataset):
    def __init__(self, folder_path, tokenizer, max_length):
        self.folder_path = folder_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._load_data()

    def _load_data(self): 
        """load json files from input folder """
        data = []
        for file_name in os.listdir(self.folder_path):
            file_path = os.path.join(self.folder_path, file_name)
            with open(file_path, 'r') as file:
                reviews = json.load(file)
            data.extend(reviews)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        instance = self.data[idx]
        
        if instance["keyword"] is None:
            instance["keyword"] = "NULL"
        
        review_tokens = self.tokenizer.encode(instance['review'], truncation=True, max_length=self.max_length)
        summary_tokens = self.tokenizer.encode(instance['keyword'], truncation=True, max_length=self.max_length)
        review_tokens = self._pad_sequence(review_tokens, self.max_length)
        summary_tokens = self._pad_sequence(summary_tokens, self.max_length)
        
        try:
            torch.tensor(summary_tokens)
        except:
            logger.warning(f"idx: {idx}; \nreview_tokens: {review_tokens};\n reviews:{instance['review']}")
 
        return {
            'input_ids': torch.tensor(review_tokens, dtype=torch.int64),
            'labels': torch.tensor(summary_tokens, dtype=torch.int64),
        }
        
    def _pad_sequence(self, sequence, max_length):
        if len(sequence) < max_length:
            #sequence = self.tokenizer.encode(self.tokenizer.pad_token * (max_length - len(sequence))) +sequence
            sequence = [self.tokenizer.pad_token_id] * (max_length - len(sequence)) +sequence
        else:
            sequence = sequence[:max_length]
        
        return sequence

    


def _train_per_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train() #no dropout so far
    total_loss = 0 
    
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = (input_ids != tokenizer.pad_token_id).type(input_ids.type())
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits        
        #loss = loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))
        loss = loss_fn(logits, labels)
        
        optimizer.zero_grad() #backward pass and optimize
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    avg_loss = total_loss/len(dataloader)
    return avg_loss
    
def _eval_per_epoch(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = (input_ids != tokenizer.pad_token_id).type(input_ids.type())

            outputs = model(input_ids=input_ids,attention_mask=attention_mask, labels=labels)
            logits = outputs.logits 
            loss = loss_fn(logits, labels)
            #loss = loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))#not customized yet
            
            total_loss += loss.item()
            
    avg_loss = total_loss / len(dataloader)
    return avg_loss 
   
def train(model, num_epochs, model_path, loss_fn, device, optimizer):
    """fine tune model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    best_val_loss = float('inf')
    for epoch in range(num_epochs):

        # Train and validate the model for one epoch
        train_loss = _train_per_epoch(model, train_dataloader, loss_fn, optimizer, device)
        val_loss = _eval_per_epoch(model, val_dataloader, loss_fn, device)
        print(f"Epoch {epoch+1}/{num_epochs} - Train loss: {train_loss} - Val loss: {val_loss}")
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Train loss: {train_loss} - Val loss: {val_loss}")
        
        # Save the model if improved val_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)


    
        
if __name__ == "__main__":
    #path args
    dataset_folder = os.path.join("dataset", "steam_indie_500_labelled")  # JSON files
    model_path = os.path.join("models", f"ts_{datetime.now()}_pt") 
    
    #tuning params
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_length = 32
    batch_size = 4
    num_workers = 2
    num_epochs = 5
    lr = 5e-5
    
    #model setup
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')  # Load the GPT-2 tokenizer
    tokenizer.pad_token = tokenizer.eos_token 
    tokenizer.pad_token_id = tokenizer.encode(tokenizer.pad_token)[0]
    
    dataset = GPT2Dataset(dataset_folder, tokenizer, max_length)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    model = GPT2LMHeadModel.from_pretrained('gpt2')  
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=50256)

    train(model, num_epochs, model_path, loss_fn, device)
   




    