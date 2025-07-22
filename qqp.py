import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class QQPDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_length=64):
        self.data = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Optionally filter out rows with null questions
        self.data = self.data.dropna(subset=['question1', 'question2']).reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        q1 = str(self.data.loc[idx, 'question1'])
        q2 = str(self.data.loc[idx, 'question2'])

        # Return raw texts; tokenization will be done in collate_fn for batching
        return {"question1": q1, "question2": q2}

def collate_fn(batch, tokenizer, max_length=64):
    questions1 = [item["question1"] for item in batch]
    questions2 = [item["question2"] for item in batch]

    enc1 = tokenizer(questions1, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
    enc2 = tokenizer(questions2, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')

    return enc1, enc2

# Usage inside main():

csv_file_path = "path/to/your/quora_question_pairs.csv"  # replace with your actual path
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

dataset = QQPDataset(csv_file_path, bert_tokenizer, max_length=64)

dataloader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=lambda batch: collate_fn(batch, bert_tokenizer, max_length=64),
)
