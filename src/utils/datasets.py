import torch
from torch.utils.data import Dataset

class GeneratedDataDataset(Dataset):
    def __init__(self, data_source, text_entry_name='C', label_entry_name='Y', idx_entry_name='idx'):
        # Initialize your dataset here
        self.load_data(data_source, text_entry_name=text_entry_name, label_entry_name=label_entry_name, idx_entry_name=idx_entry_name)

    def __len__(self):
        # Return the total number of samples
        return len(self.text)

    def __getitem__(self, idx):
        # Return a specific sample
        text, label, index = [], [], []
        for _i in idx:
            text.append(self.text[_i])
            label.append(self.label[_i])
            index.append(self.index[_i])
        # # Process your sample data as needed (e.g., transform, preprocess)
        # processed_sample = self.process_sample(sample)
        # return processed_sample
        # print(text, label, index)
        return text, label, index

    def load_data(self, data_source, text_entry_name='C', label_entry_name='Y', idx_entry_name='idx'):
        # Return a list of samples, where each sample is a dictionary or tuple
        # containing the data you need for each sample.
        self.text = []
        self.label = []
        self.index = []
        for data in data_source:
            self.text.append(data[text_entry_name])
            self.label.append(data[label_entry_name])
            self.index.append(data[idx_entry_name])
        # Load data from files, preprocess, and add to the data list
        # Example: data.append({'text': 'sample text', 'label': 0})
        # return data

    # def process_sample(self, sample):
    #     # Process and transform your sample data as needed
    #     # Example: tokenization, one-hot encoding, or image transformations
    #     processed_sample = {'text': sample['text'], 'label': sample['label']}
    #     return processed_sample