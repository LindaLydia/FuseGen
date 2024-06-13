from datasets import load_metric, load_dataset, load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
import os
from .glue_processor import GLUEProcessor

from utils.constant import MODEL_PATH


class IMDbProcessor(GLUEProcessor):
    def __init__(self, task_name, model_name, model_ckpt, output_dir, device, **train_args):
        super().__init__(task_name, model_name, model_ckpt, output_dir, device, **train_args)

    def load_model(self):
        self.num_labels = 2
        self.is_regression = False
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH[self.model_name])
        load_name = self.model_ckpt if self.model_ckpt is not None else self.model_name
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH[load_name], num_labels=self.num_labels).to(self.device)
        self.data_collator = DataCollatorWithPadding(self.tokenizer)
        self.sentence1_key, self.sentence2_key = 'text', None

    def load_dataset(self):
        data_path = f'data/imdb'
        # if os.path.exists(data_path):
        #     self.dataset = load_from_disk(data_path)
        # else:
        #     self.dataset = load_dataset('imdb')
        #     self.dataset.save_to_disk(data_path)

        # for name, subset in self.dataset.items():
        #     self.dataset[name] = subset.add_column('idx', list(range(len(subset))))
        # # imdb don't have a val set, thus the reported metric is on test set
        # self.dataset[self.validation_key] = self.dataset[self.test_key]

        self.dataset = None

        self.sentence1_key, self.sentence2_key = 'text', None
        self.encoded_dataset = None
        # self.encoded_dataset = self._encode_dataset(self.dataset)
        # self.metric = load_metric("glue", "sst2")
        self.metric = None
        self.main_metric_name = "eval_accuracy"

    def preprocess_function(self, examples):
        examples[self.sentence1_key] = [x.replace('<br />', '\n') for x in examples[self.sentence1_key]]
        return self.tokenizer(examples[self.sentence1_key], truncation=True, max_length=512)


class IMDbProcessor_evaluator(GLUEProcessor):
    def __init__(self, task_name, model_name, model_ckpt, output_dir, device, **train_args):
        super().__init__(task_name, model_name, model_ckpt, output_dir, device, **train_args)

    def load_model(self):
        self.num_labels = 2
        self.is_regression = False
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH[self.model_name])
        load_name = self.model_ckpt if self.model_ckpt is not None else self.model_name
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH[load_name], num_labels=self.num_labels).to(self.device)
        self.data_collator = DataCollatorWithPadding(self.tokenizer)

    def load_dataset(self):
        data_path = f'data/imdb'
        # if os.path.exists(data_path):
        #     self.dataset = load_from_disk(data_path)
        # else:
        #     self.dataset = load_dataset('imdb')
        #     self.dataset.save_to_disk(data_path)

        # for name, subset in self.dataset.items():
        #     self.dataset[name] = subset.add_column('idx', list(range(len(subset))))
        # # imdb don't have a val set, thus the reported metric is on test set
        # self.dataset[self.validation_key] = self.dataset[self.test_key]

        self.dataset = None

        self.sentence1_key, self.sentence2_key = 'text', None
        self.encoded_dataset = None
        # self.encoded_dataset = self._encode_dataset(self.dataset)
        # self.metric = load_metric("glue", "sst2")
        self.metric = None
        self.main_metric_name = "eval_accuracy"

    def preprocess_function(self, examples):
        examples[self.sentence1_key] = [x.replace('<br />', '\n') for x in examples[self.sentence1_key]]
        return self.tokenizer(examples[self.sentence1_key], truncation=True, max_length=512)



class YelpProcessor(GLUEProcessor):
    def __init__(self, task_name, model_name, model_ckpt, output_dir, device, **train_args):
        super().__init__(task_name, model_name, model_ckpt, output_dir, device, **train_args)

    def load_model(self):
        self.num_labels = 2
        self.is_regression = False
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH[self.model_name])
        load_name = self.model_ckpt if self.model_ckpt is not None else self.model_name
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH[load_name], num_labels=self.num_labels).to(self.device)
        self.data_collator = DataCollatorWithPadding(self.tokenizer)
        self.sentence1_key, self.sentence2_key = 'text', None

    def load_dataset(self):
        data_path = f'data/yelp'
        # if os.path.exists(data_path):
        #     self.dataset = load_from_disk(data_path)
        # else:
        #     self.dataset = load_dataset('imdb')
        #     self.dataset.save_to_disk(data_path)

        # for name, subset in self.dataset.items():
        #     self.dataset[name] = subset.add_column('idx', list(range(len(subset))))
        # # imdb don't have a val set, thus the reported metric is on test set
        # self.dataset[self.validation_key] = self.dataset[self.test_key]

        self.dataset = None

        self.sentence1_key, self.sentence2_key = 'text', None
        self.encoded_dataset = None
        # self.encoded_dataset = self._encode_dataset(self.dataset)
        # self.metric = load_metric("glue", "sst2")
        self.metric = None
        self.main_metric_name = "eval_accuracy"

    def preprocess_function(self, examples):
        examples[self.sentence1_key] = [x.replace('<br />', '\n') for x in examples[self.sentence1_key]]
        return self.tokenizer(examples[self.sentence1_key], truncation=True, max_length=512)

