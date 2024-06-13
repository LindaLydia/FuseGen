import torch
# from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset
import json
import random

# Load and preprocess data from the jsonl file
class TokenizedDataset(Dataset):
    def __init__(self, file_path='', text_column='text', label_column='label', index_column='idx', tokenizer=None, max_length=512, device='cpu', max_sample=-1, small_dataset_shuffle=True):
        self.text = []
        self.ids = []
        self.attention_mask = []
        self.label = []
        self.idx = []
        if file_path == '':
            self.ids = torch.tensor([self.ids],dtype=torch.int64).to(device)
            self.attention_mask = torch.tensor([self.attention_mask],dtype=torch.int64).to(device)
            self.label = torch.tensor(self.label,dtype=torch.int64).to(device)
            self.idx = torch.tensor(self.idx,dtype=torch.int64).to(device)
        else:
            if 'imdb' in file_path and (0 < max_sample < 1000) and small_dataset_shuffle==True:
                lines = []
                with open(file_path, 'r') as file:
                    for line in file:
                        lines.append(line)
                random.shuffle(lines)
                counter = 0
                for line in lines:
                    item = json.loads(line.strip())
                    text = item[text_column]
                    label = item[label_column]  # Assuming your jsonl file contains a 'label' field
                    # idx = item[index_column]
                    tokenized = tokenizer.encode_plus(
                        text,
                        add_special_tokens=True,
                        max_length=max_length,
                        padding='max_length',
                        truncation=True,
                        return_attention_mask=True,
                        return_tensors='pt',
                    )
                    self.text.append(text)
                    self.ids.append(tokenized['input_ids'])
                    self.attention_mask.append(tokenized['attention_mask'])
                    self.label.append(label)
                    self.idx.append(counter) # append counter this time, not the original idx
                    counter += 1
                    if counter == max_sample:
                        break
            else:
                with open(file_path, 'r') as file:
                    counter = 0
                    for line in file:
                        item = json.loads(line.strip())
                        text = item[text_column]
                        label = item[label_column]  # Assuming your jsonl file contains a 'label' field
                        idx = item[index_column]
                        tokenized = tokenizer.encode_plus(
                            text,
                            add_special_tokens=True,
                            max_length=max_length,
                            padding='max_length',
                            truncation=True,
                            return_attention_mask=True,
                            return_tensors='pt',
                        )
                        self.text.append(text)
                        self.ids.append(tokenized['input_ids'])
                        self.attention_mask.append(tokenized['attention_mask'])
                        self.label.append(label)
                        self.idx.append(idx)
                        counter += 1
                        if max_sample > 0 and counter == max_sample:
                            break
            # print("in TokenizedDataset init", self.text[0], self.ids[0], self.attention_mask[0], self.label[0], self.idx[0])
            # print("in TokenizedDataset init", self.text[-1], self.ids[-1], self.attention_mask[-1], self.label[-1], self.idx[-1])
            # print(self.ids)
            # print(self.label)
            # print(self.ids[-1].dtype)
            # self.ids = torch.stack(self.ids).squeeze().to(device)
            # self.attention_mask = torch.stack(self.attention_mask).squeeze().to(device)
            # self.label = torch.tensor(self.label).long().to(device)
            # self.idx = torch.tensor(self.idx).long().to(device)
            self.ids = torch.stack(self.ids).squeeze()
            self.attention_mask = torch.stack(self.attention_mask).squeeze()
            self.label = torch.tensor(self.label).long()
            self.idx = torch.tensor(self.idx).long()
        # print(self.ids.shape, self.attention_mask.shape, self.label.shape, self.idx.shape)
        # print(self.ids.dtype, self.attention_mask.dtype, self.label.dtype, self.idx.dtype)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        return self.ids[index], self.attention_mask[index], self.label[index], self.idx[index]


# Load and preprocess data from the jsonl file
class TokenizedQADataset(Dataset):
    # {"context": "He survived this on Day 47, but was evicted on Day 52 through the backdoor after receiving only 1.52% of the overall final vote to win.", "question": "How many days did joshuah alagappan survive in big brother?", "answers": {"answer_start": [20], "text": ["Day 47"]}, "id": "ID_1982-0-0"}
    def __init__(self, file_path='', context_column='context', question_column='question', label_column='answers', index_column='idx', tokenizer=None, max_length=512, device='cpu', max_sample=-1, small_dataset_shuffle=True):
        self.context = []
        self.question = []
        self.ids = []
        self.attention_mask = []
        self.offset_mapping = []
        self.sample_mapping = []
        self.label = []
        self.idx = []
        if file_path == '':
            self.ids = torch.tensor([self.ids],dtype=torch.int64).to(device)
            self.attention_mask = torch.tensor([self.attention_mask],dtype=torch.int64).to(device)
            self.offset_mapping = torch.tensor([self.offset_mapping],dtype=torch.int64).to(device)
            self.sample_mapping = torch.tensor([self.sample_mapping],dtype=torch.int64).to(device)
            # self.label = torch.tensor(self.label,dtype=torch.int64).to(device)
            self.idx = torch.tensor(self.idx,dtype=torch.int64).to(device)
        else: 
            with open(file_path, 'r') as file:
                counter = 0
                for line in file:
                    item = json.loads(line.strip())
                    context = item[context_column]
                    question = item[question_column]
                    label = item[label_column]  # Assuming your jsonl file contains a 'label' field
                    idx = item[index_column]
                    tokenized = tokenizer.encode_plus(
                        question,
                        context,
                        add_special_tokens=True,
                        max_length=max_length,
                        padding='max_length',
                        truncation="only_second",
                        return_attention_mask=True,
                        return_tensors='pt',
                        return_overflowing_tokens=True,
                        return_offsets_mapping=True,
                    )
                    self.context.append(context)
                    self.question.append(question)
                    self.ids.append(tokenized['input_ids'])
                    # self.ids.append(tokenized)
                    self.attention_mask.append(tokenized['attention_mask'])
                    self.offset_mapping.append(tokenized["offset_mapping"])
                    self.sample_mapping.append(tokenized["overflow_to_sample_mapping"])
                    self.label.append(label)
                    self.idx.append(idx)
                    counter += 1
                    if max_sample > 0 and counter == max_sample:
                        break
            # self.ids = torch.stack(self.ids).squeeze()
            # self.attention_mask = torch.stack(self.attention_mask).squeeze()
            # self.offset_mapping = torch.stack(self.offset_mapping).squeeze()
            # self.sample_mapping = torch.stack(self.sample_mapping).squeeze()
            # # self.label = torch.tensor(self.label).long()
            self.idx = torch.tensor(self.idx).long()
        # print(self.ids.shape, self.attention_mask.shape, self.label.shape, self.idx.shape)
        # print(self.ids.dtype, self.attention_mask.dtype, self.label.dtype, self.idx.dtype)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        return self.context[index], self.ids[index], self.attention_mask[index], self.offset_mapping[index], self.sample_mapping[index], self.label[index], self.idx[index]

