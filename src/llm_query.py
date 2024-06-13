# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script can be used to generate datasets.
"""

import argparse
import json
import os

import torch
import wandb
import datasets
import jsonlines
import random

from tasks import *
from utils.cls_generator import DataGenerator, EvaluationGenerator, ClassificationGenerator, C_KEY
from utils.qa_generator import QADataGenerator
from utils.generation import LLMWrapper, SimpleLLM
from utils.basic_utils import init_logging, set_seed, read_jsonl, save_jsonl, save_jsonl_append, modify_idx_in_json_file
from utils.constant import PROMPTS


def task2processor(task_name):
    if task_name == 'imdb':
        return IMDbProcessor
    elif task_name == 'yelp':
        return YelpProcessor
    elif task_name == 'sst-2':
        return SST2Processor
    elif task_name == 'squad' or task_name == 'adversarial_qa':
        return QAProcessor
    else:
        return GLUEProcessor


def create_output_name(args):
    name = [args.gen_model_name, f"topk{args.gen_top_k}", f"topp{args.gen_top_p}", args.gen_task_file.split('/')[-1][:-5]]

    if args.gen_decay_constant > 0:
        name.append(f"self-debias-{args.gen_decay_constant}")
    return '_'.join(name)


def gen_syn_data_few_shot(args):
    '''
        # python main.py --task_file tasks/yelp/yelp-x2-2.json 
                         --output_dir yelp/output/50k/ 
                         --model_name chatglm3-6b-base 
                         --small_model_name distilbert-base-uncased 
                         --min_length 10 --max_length 100 
                         --top_k 0 --top_p 0.9 --decay_constant 200 
                         --batch_size 20 --train_batch_size 32 
                         --learning_rate 2e-5 --num_entries_per_input 50000
    '''
    if args.task_name in ['mnli', 'mnliMisM', 'qnli', 'squad']:
        args.gen_input_file = './utils/wiki_data/wiki_short.jsonl'
    else:
        args.gen_input_file = None
    hasattr(args, 'gen_output_dir')# "The output directory to which the generated dataset is saved"
    hasattr(args, 'gen_task_file') # "A json file providing the instructions and other information required for dataset generation. "
    hasattr(args, 'gen_input_file') # "An input file containing the generated data"
    hasattr(args, 'gen_model_name') # "The pretrained model to use for dataset generation. Currently, only variants of GPT2 are supported."
    hasattr(args, 'gen_batch_size') # "The batch size for generation (only if --input_file is not set)"
    hasattr(args, 'gen_num_entries_per_input')
    hasattr(args, 'gen_max_length')
    hasattr(args, 'gem_min_length')
    args.gen_top_p = 0.9
    args.gen_top_k = 0
    args.gen_temperature = 1.0
    args.gen_decay_constant = 200
    args.gen_log_every = 10000
    # hasattr(args, 'output_dir')
    with open(args.gen_task_file, 'r', encoding='utf8') as fh:
        task_specification = json.load(fh)
    args.task_specification = task_specification
    args.task_name = task_specification["task_name"]
    is_stage_two = task_specification['stage'] == 'x2'
    zero_shot = task_specification['stage'] == 'zs'

    # if is_stage_two:
    #     output_name = create_output_name(args)
    #     args.gen_output_dir = os.path.join(args.gen_output_dir, output_name)
    #     # wandb.init(project=os.getenv("WANDB_PROJECT"), entity=os.getenv("WANDB_ENTITY"), config=args, name=output_name,
    #     #            tags=[task_specification["task_name"]])

    logging = init_logging(log_file=args.gen_output_dir + '/output.log', stdout=True)
    # logging.info(f"Parameters: {args}")

    # args_file = os.path.join(args.gen_output_dir, f'{task_specification["task_name"]}-args.json')
    # with open(args_file, 'w', encoding='utf8') as fh:
    #     fh.write(json.dumps(vars(args), indent=4))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = task2processor(args.task_name)(task_name=args.task_name,
                                               model_name='distilbert-base-uncased',
                                               model_ckpt=None,
                                               output_dir=args.gen_output_dir,
                                               device=device,
                                               num_epochs=3,
                                               train_batch_size=32,
                                               learning_rate=2e-5
                                               ) # the parameter are indeed useless, just do not want to remove them
    # processor = None

    logging.info("Building model...")
    model = LLMWrapper(model_name=args.gen_model_name, use_cuda=True, device=args.device, args=args)

    logging.info("Building generator...")
    if isinstance(processor, QAProcessor):
        generator = QADataGenerator(
            task_spec=task_specification, model=model, max_length=args.gen_max_length, min_length=args.gen_min_length,
            top_p=args.gen_top_p, top_k=args.gen_top_k, temperature=args.gen_temperature,
            # processor=None, 
            processor=processor, 
            do_sample=True, seed=args.seed, output_dir=args.gen_output_dir
        )
        if zero_shot:
            logging.info("Starting inference under zero-shot setting...")
            generator.zero_shot_inference(args.gen_batch_size)
        elif is_stage_two:
            logging.info("Starting dataset generation, stage two...")
            # inputs = datasets.load_from_disk(args.gen_input_file)
            # dataset = generator.generate_question(inputs, num_entries_per_input=args.gen_num_entries_per_input,
            #                                       batch_size=args.gen_batch_size, log_every=args.gen_log_every)
            # dataset.save_to_disk(args.gen_output_dir)
            inputs = [i[C_KEY] for i in read_jsonl(args.gen_input_file)]
            random.shuffle(inputs)
            inputs = inputs[:int(args.gen_num_entries_per_input*3*(2 if 'glm' in args.gen_model_name else 1))]
            dataset = generator.generate_answer_ner(inputs)
            # dataset.save_to_disk(args.output_dir)
            # inputs = datasets.load_from_disk(args.input_file)
            outputs = generator.generate_question(dataset, num_entries_per_input=1,
                                                  batch_size=args.gen_batch_size, log_every=args.gen_log_every)
            # dataset.save_to_disk(args.output_dir)
            # assert len(outputs) >= args.gen_num_entries_per_input, f"Error, requiring {args.gen_num_entries_per_input} samples, but only generated {len(outputs)} samples"
            while len(outputs) < args.gen_num_entries_per_input:
                outputs = outputs + outputs[:min(len(outputs), args.gen_num_entries_per_input-len(outputs))]
            assert len(outputs) >= args.gen_num_entries_per_input, f"Error, requiring {args.gen_num_entries_per_input} samples, but only generated & copied only {len(outputs)} samples"
            logging.info(f"Dataset generation complete, dataset contains {len(outputs)} entries")
            random.shuffle(outputs)
            # dataset_path = os.path.join(args.gen_output_dir, f'{task_specification["task_name"]}-dataset.jsonl')
            for sample_file_name in ['train_noflip', 'train']:
                dataset_path = os.path.join(args.gen_output_dir, f'{sample_file_name}.jsonl')
                save_jsonl_append(outputs[:args.gen_num_entries_per_input], dataset_path)
            logging.info(f"Done saving dataset to file '{dataset_path}'")
        else:
            logging.info("Starting dataset generation, stage one...")
            dataset = generator.generate_answer_ner()
            dataset.save_to_disk(args.gen_output_dir)
    else:
        generator = DataGenerator(
            task_spec=task_specification, model=model, max_length=args.gen_max_length,
            top_p=args.gen_top_p, top_k=args.gen_top_k, temperature=args.gen_temperature, do_sample=True,
            # processor=None,
            processor=processor,
            min_length=args.gen_min_length,
            is_stage_two=is_stage_two,
            decay_constant=args.gen_decay_constant,
            output_dir=args.gen_output_dir
        )

        if zero_shot:
            logging.info("Starting inference under zero-shot setting...")
            dataset = processor.dataset[processor.validation_key]
            generator.zero_shot_inference(dataset, args.gen_batch_size)
        else:
            if args.gen_input_file:
                logging.info(f"Use condition c from {args.gen_input_file}")
                inputs = [i[C_KEY] for i in read_jsonl(args.gen_input_file)]
                if 'nli' in args.task_name:
                    inputs = [i[C_KEY] for i in read_jsonl(args.gen_input_file)]
                    random.shuffle(inputs)
                    inputs = inputs[:int(args.gen_num_entries_per_input*args.num_classes*(1 if 'glm' in args.gen_model_name else 0.5))]
            elif is_stage_two and processor.sentence2_key is not None:
                logging.info("Use condition c from dataset")
                inputs = processor.dataset[processor.train_key][processor.sentence1_key]
            else:
                logging.info("Do not use condition c")
                inputs = None
            
            logging.info("Starting dataset generation...")
            outputs = generator.generate_dataset(inputs, num_entries_per_input=(1 if 'nli' in args.task_name else args.gen_num_entries_per_input),
                                                 batch_size=args.gen_batch_size, log_every=args.gen_log_every, task_name=args.task_name)

            # assert len(outputs) >= args.gen_num_entries_per_input, f"Error, requiring {args.gen_num_entries_per_input} samples, but only generated {len(outputs)} samples"
            while len(outputs) < args.gen_num_entries_per_input:
                outputs = outputs + outputs[:min(len(outputs), args.gen_num_entries_per_input-len(outputs))]
            assert len(outputs) >= args.gen_num_entries_per_input, f"Error, requiring {args.gen_num_entries_per_input} samples, but only generated & copied only {len(outputs)} samples"
            logging.info(f"Dataset generation complete, dataset contains {len(outputs)} entries")
            random.shuffle(outputs)
            # dataset_path = os.path.join(args.gen_output_dir, f'{task_specification["task_name"]}-dataset.jsonl')
            for sample_file_name in ['train_noflip', 'train']:
                dataset_path = os.path.join(args.gen_output_dir, f'{sample_file_name}.jsonl')
                save_jsonl_append(outputs[:args.gen_num_entries_per_input], dataset_path)
            logging.info(f"Done saving dataset to file '{dataset_path}'")

    # if is_stage_two:
    #     wandb.save(args.gen_output_dir)
    for sample_file_name in ['train_noflip', 'train']:
        save_path = os.path.join(args.gen_output_dir, f'{sample_file_name}.jsonl')
        modify_idx_in_json_file(dataset_path, save_path)


def gen_evaluation(args, data_file_name='data_file'):
# def gen_evaluation(args, query_data=None, data_file_name='data_file'):
    hasattr(args, 'query_output_dir')# "The output directory to which the generated dataset is saved"
    hasattr(args, 'query_task_file') # "A json file providing the instructions and other information required for dataset generation. "
    hasattr(args, 'query_input_file') # "An input file containing generated text pairs by the other model."
    hasattr(args, 'query_model_name') # "The pretrained model to use for evaluation."
    hasattr(args, 'query_batch_size') # "The batch size for evaluation"
    # hasattr(args, 'query_num_entries_per_input')
    hasattr(args, 'query_max_length')
    hasattr(args, 'query_min_length')
    args.query_top_p = 1.0
    args.query_top_k = 0 # top 1 actually
    args.query_temperature = 0.7
    args.query_decay_constant = 0
    args.query_log_every = 10000
    # hasattr(args, 'output_dir')
    with open(args.query_task_file, 'r', encoding='utf8') as fh:
        task_specification = json.load(fh)
    args.task_specification = task_specification
    args.task_name = task_specification["task_name"]

    assert task_specification['stage'] == 'x3', "sementic prediction stage should be stage 3"
    if 'judge' not in args.query_output_dir:
        output_name = 'judge__' + args.query_model_name + "__" + data_file_name.replace('/','_')[:-6]
        args.query_output_dir = os.path.join(args.query_output_dir, output_name)
    # wandb.init(project=os.getenv("WANDB_PROJECT"), entity=os.getenv("WANDB_ENTITY"), config=args, name=output_name,
    #            tags=[task_specification["task_name"]])

    # input_samples = jsonlines.open(self.input_dir, 'r')

    logging = init_logging(log_file=args.query_output_dir + '/output.log', stdout=True)
    # logging.info(f"Parameters: {args}")

    args_file = os.path.join(args.query_output_dir, f'{task_specification["task_name"]}-args.json')
    with open(args_file, 'w', encoding='utf8') as fh:
        fh.write(json.dumps(vars(args), indent=4))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # processor = task2processor(args.task_name)(task_name=args.task_name,
    #                                            model_name='distilbert-base-uncased',
    #                                            model_ckpt=None,
    #                                            output_dir=args.query_output_dir,
    #                                            device=device,
    #                                            num_epochs=3,
    #                                            train_batch_size=32,
    #                                            learning_rate=2e-5
    #                                            ) # the parameter are indeed useless, just do not want to remove them
    processor = None

    logging.info("Building model...")
    model = SimpleLLM(model_name=args.query_model_name, use_cuda=True, args=args)

    task_query_prompting = PROMPTS[args.task_name]
    # print(f"task_query_prompting=***{task_query_prompting}***")

    # if query_data==None:
    if data_file_name=='data_file':
        input_reader = jsonlines.open((args.query_input_file), 'r')
        args.query_input_samples = [line for line in input_reader]
    else:
        raw_train_data_path = data_file_name
        print(f"raw_train_data_path={raw_train_data_path}")
        raw_train_reader = jsonlines.open(raw_train_data_path, 'r')
        raw_train_samples = [line for line in raw_train_reader]
        query_data = []
        for indice in args.query_indices:
            query_data.append(raw_train_samples[indice])
            # query_data.append({'C': train_data[indice].text, 'Y': train_data[indice].label, 'idx': train_data[indice].idx})
            # print(f"[debug] query_data[-1]={query_data[-1]}, {query_data[-1]['C']}")        
        args.query_input_samples = query_data

    logging.info("Building generator...")
    if args.task_name == 'squad' or args.task_name == 'adversarial_qa':
        # generator = QADataGenerator(
        #     task_spec=task_specification, model=model, max_length=args.query_max_length, min_length=args.query_min_length,
        #     top_p=args.query_top_p, top_k=args.query_top_k, temperature=args.query_temperature,
        #     # processor=None, 
        #     processor=processor, 
        #     do_sample=True, seed=args.seed, output_dir=args.query_output_dir
        # )
        # if zero_shot:
        #     logging.info("Starting inference under zero-shot setting...")
        #     generator.zero_shot_inference(args.query_batch_size)
        # elif is_stage_two:
        #     logging.info("Starting dataset generation, stage two...")
        #     inputs = datasets.load_from_disk(args.query_input_file)
        #     dataset = generator.generate_question(inputs, num_entries_per_input=args.query_num_entries_per_input,
        #                                           batch_size=args.query_batch_size, log_every=args.query_log_every)
        #     dataset.save_to_disk(args.query_output_dir)
        # else:
        #     logging.info("Starting dataset generation, stage one...")
        #     dataset = generator.generate_answer_ner()
        #     dataset.save_to_disk(args.query_output_dir)
        pass
    else:
        generator = EvaluationGenerator(
            task_spec=task_specification, model=model, max_length=args.query_max_length,
            top_p=args.query_top_p, top_k=args.query_top_k, temperature=args.query_temperature, do_sample=True,
            # processor=None,
            processor=processor,
            min_length=args.query_min_length,
            decay_constant=args.query_decay_constant,
            output_dir=args.query_output_dir,
            # input_samples=args.query_input_samples,
        )
        # if args.query_input_file:
        #     logging.info(f"Use condition c from {args.query_input_file}")
        #     inputs = [i[C_KEY] for i in read_jsonl(args.query_input_file)]
        # elif is_stage_two and processor.sentence2_key is not None:
        #     logging.info("Use condition c from dataset")
        #     inputs = processor.dataset[processor.train_key][processor.sentence1_key]
        # else:
        #     logging.info("Do not use condition c")
        #     inputs = None
        inputs = None

        result_data = []
        QUERY_BATCH = args.query_batch_size
        _BATCH_BIN = [i for i in range(0,len(args.query_input_samples),QUERY_BATCH)] + [len(args.query_input_samples)]
        for batch_bin_index in range(len(_BATCH_BIN)-1):
            # result_data += gen_evaluation(args, query_data=args.query_input_samples[_BATCH_BIN[batch_bin_index]:_BATCH_BIN[batch_bin_index+1]])
            logging.info("Starting evaluation generation...")
            result_data += generator.evaluate(input_samples=args.query_input_samples[_BATCH_BIN[batch_bin_index]:_BATCH_BIN[batch_bin_index+1]], num_entries_per_input=args.query_num_entries_per_input,
                                            batch_size=args.query_batch_size, log_every=args.query_log_every, task_chat_prompt=task_query_prompting)

        logging.info(f"Evaluation complete, contains evaluation of {len(result_data)} entries")
        dataset_path = os.path.join(args.query_output_dir, f'evaluation.jsonl')
        # with open(dataset_path, "a") as file:
        #     for entry in result_data:
        #         json.dump(entry, file)
        #         file.write("\n")
        save_jsonl(result_data, dataset_path)
        logging.info(f"Done saving dataset to file '{dataset_path}'")

    return result_data


def gen_model_golden_evaluation(args, data_file_name='data_file'):
    # task_prompt = {
    #     'imdb': """Analyze the following movie review and determine if the sentiment is: positive or negative. Return answer in single word as either positive or negative: {}""",
    #     "yelp": """Analyze the following restaurant review and determine if the sentiment is: positive or negative. Return answer in single word as either positive or negative: {}""",
    # }
    task_prompt = {
        'imdb': """{} The sentiment of the movie review is (answer in single word as either positive or negative): """,
        "yelp": """{} The sentiment of the restaurant review is (answer in single word as either positive or negative): """,
        "qnli": """The context sentence is: {} The question is: {} The answer to the question is contained in the context (anser in singel word "yes" or "no"): """
    }
    hasattr(args, 'query_output_dir')# "The output directory to which the generated dataset is saved"
    hasattr(args, 'query_input_file') # "An input file containing generated text pairs by the other model."
    hasattr(args, 'query_model_name') # "The pretrained model to use for evaluation."
    hasattr(args, 'query_batch_size') # "The batch size for evaluation"
    # hasattr(args, 'query_num_entries_per_input')
    hasattr(args, 'query_max_length')
    hasattr(args, 'query_min_length')
    args.query_max_length = 100
    args.query_top_p = 1.0
    args.query_top_k = 0 # top 1 actually
    args.query_temperature = 0.7
    args.query_decay_constant = 0
    args.query_log_every = 10000
    # hasattr(args, 'output_dir')
    task_specification = {
        "task_name": args.task_name,
        "stage": 'x3',
        "task_prompt": task_prompt[args.task_name],
    }
    args.task_specification = task_specification

    assert task_specification['stage'] == 'x3', "sementic prediction stage should be stage 3"
    if 'judge' not in args.query_output_dir:
        output_name = 'judge__' + args.query_model_name + "__" + data_file_name.replace('/','_')[:-6]
        args.query_output_dir = os.path.join(args.query_output_dir, output_name)
    # wandb.init(project=os.getenv("WANDB_PROJECT"), entity=os.getenv("WANDB_ENTITY"), config=args, name=output_name,
    #            tags=[task_specification["task_name"]])

    # input_samples = jsonlines.open(self.input_dir, 'r')

    logging = init_logging(log_file=args.query_output_dir + f'/{args.query_model_name}-output.log', stdout=True)
    # logging.info(f"Parameters: {args}")

    args_file = os.path.join(args.query_output_dir, f'{task_specification["task_name"]}-args.json')
    with open(args_file, 'w', encoding='utf8') as fh:
        fh.write(json.dumps(vars(args), indent=4))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # processor = task2processor(args.task_name)(task_name=args.task_name,
    #                                            model_name='distilbert-base-uncased',
    #                                            model_ckpt=None,
    #                                            output_dir=args.query_output_dir,
    #                                            device=device,
    #                                            num_epochs=3,
    #                                            train_batch_size=32,
    #                                            learning_rate=2e-5
    #                                            ) # the parameter are indeed useless, just do not want to remove them
    processor = None

    logging.info("Building model...")
    model = SimpleLLM(model_name=args.query_model_name, use_cuda=True, args=args)

    task_query_prompting = PROMPTS[args.task_name]
    # print(f"task_query_prompting=***{task_query_prompting}***")

    # if query_data==None:
    if data_file_name=='data_file':
        input_reader = jsonlines.open((args.query_input_file), 'r')
        args.query_input_samples = [line for line in input_reader]
    else:
        raw_train_data_path = data_file_name
        print(f"raw_train_data_path={raw_train_data_path}")
        raw_train_reader = jsonlines.open(raw_train_data_path, 'r')
        raw_train_samples = [line for line in raw_train_reader]
        query_data = []
        for indice in args.query_indices:
            query_data.append(raw_train_samples[indice])
            # query_data.append({'C': train_data[indice].text, 'Y': train_data[indice].label, 'idx': train_data[indice].idx})
            # print(f"[debug] query_data[-1]={query_data[-1]}, {query_data[-1]['C']}")        
        args.query_input_samples = query_data

    logging.info("Building generator...")
    if args.task_name == 'squad' or args.task_name == 'adversarial_qa':
        # generator = QADataGenerator(
        #     task_spec=task_specification, model=model, max_length=args.query_max_length, min_length=args.query_min_length,
        #     top_p=args.query_top_p, top_k=args.query_top_k, temperature=args.query_temperature,
        #     # processor=None, 
        #     processor=processor, 
        #     do_sample=True, seed=args.seed, output_dir=args.query_output_dir
        # )
        # if zero_shot:
        #     logging.info("Starting inference under zero-shot setting...")
        #     generator.zero_shot_inference(args.query_batch_size)
        # elif is_stage_two:
        #     logging.info("Starting dataset generation, stage two...")
        #     inputs = datasets.load_from_disk(args.query_input_file)
        #     dataset = generator.generate_question(inputs, num_entries_per_input=args.query_num_entries_per_input,
        #                                           batch_size=args.query_batch_size, log_every=args.query_log_every)
        #     dataset.save_to_disk(args.query_output_dir)
        # else:
        #     logging.info("Starting dataset generation, stage one...")
        #     dataset = generator.generate_answer_ner()
        #     dataset.save_to_disk(args.query_output_dir)
        pass
    else:
        generator = ClassificationGenerator(
            task_spec=task_specification, model=model, max_length=args.query_max_length,
            top_p=args.query_top_p, top_k=args.query_top_k, temperature=args.query_temperature, do_sample=True,
            # processor=None,
            processor=processor,
            min_length=args.query_min_length,
            decay_constant=args.query_decay_constant,
            output_dir=args.query_output_dir,
            # input_samples=args.query_input_samples,
        )
        # if args.query_input_file:
        #     logging.info(f"Use condition c from {args.query_input_file}")
        #     inputs = [i[C_KEY] for i in read_jsonl(args.query_input_file)]
        # elif is_stage_two and processor.sentence2_key is not None:
        #     logging.info("Use condition c from dataset")
        #     inputs = processor.dataset[processor.train_key][processor.sentence1_key]
        # else:
        #     logging.info("Do not use condition c")
        #     inputs = None
        inputs = None

        result_data = []
        dataset_path = os.path.join(args.query_output_dir, f'{args.query_model_name}-evaluation.jsonl')
        QUERY_BATCH = args.query_batch_size
        _BATCH_BIN = [i for i in range(0,len(args.query_input_samples),QUERY_BATCH)] + [len(args.query_input_samples)]
        for batch_bin_index in range(len(_BATCH_BIN)-1):
            # if batch_bin_index <= 5611:
            #     continue
            # result_data += gen_evaluation(args, query_data=args.query_input_samples[_BATCH_BIN[batch_bin_index]:_BATCH_BIN[batch_bin_index+1]])
            logging.info("Starting evaluation generation...")
            result_data = generator.evaluate(input_samples=args.query_input_samples[_BATCH_BIN[batch_bin_index]:_BATCH_BIN[batch_bin_index+1]], num_entries_per_input=args.query_num_entries_per_input,
                                            batch_size=args.query_batch_size, log_every=args.query_log_every, task_chat_prompt=task_query_prompting)

            logging.info(f"Evaluation complete, contains evaluation of {len(result_data)} entries")
            # with open(dataset_path, "a") as file:
            #     for entry in result_data:
            #         json.dump(entry, file)
            #         file.write("\n")
            save_jsonl_append(result_data, dataset_path)
            logging.info(f"Done saving dataset to file '{dataset_path}'")

    return result_data



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--gen_output_dir", type=str, default=None, #required=True,
                        help="The output directory to which the generated dataset is saved")
    parser.add_argument("--gen_task_file", type=str, default=None, #required=True,
                        help="A json file providing the instructions and other information required for dataset generation. ")

    # Dataset and prompt parameters
    parser.add_argument("--gen_input_file", type=str, default=None,
                        help="An optional input file containing raw texts. This is required for generating text pair datasets.")

    # Text generation and sampling parameters
    parser.add_argument("--gen_model_name", type=str, default="gpt2-xl",
                        help="The pretrained model to use for dataset generation. Currently, only variants of GPT2 are supported.")
    parser.add_argument("--gen_batch_size", type=int, default=None,
                        help="The batch size for generation (only if --input_file is not set)")
    parser.add_argument("--gen_num_entries_per_input", type=int, default=None,
                        help="The number of entries to generate for each label (only if --input_file is not set)")
    parser.add_argument("--gen_max_length", type=int, default=40,
                        help="The maximum output length for each generated text.")
    parser.add_argument("--gen_min_length", type=int, default=1,
                        help="Min length of generated text.")
    

    # Required parameters
    parser.add_argument("--query_output_dir", type=str, #required=True,
                        help="The output directory to which the generated dataset is saved")
    parser.add_argument("--query_task_file", type=str, #required=True,
                        help="A json file providing the instructions and other information required for dataset generation. ")

    # Dataset and prompt parameters
    parser.add_argument("--query_input_file", type=str, default=None,
                        help="An optional input file containing raw texts. This is required for generating text pair datasets.")

    # Text generation and sampling parameters
    parser.add_argument("--query_model_name", type=str, default="gpt2-xl",
                        help="The pretrained model to use for dataset generation. Currently, only variants of GPT2 are supported.")
    parser.add_argument("--query_batch_size", type=int, default=None,
                        help="The batch size for generation (only if --input_file is not set)")
    parser.add_argument("--query_num_entries_per_input", type=int, default=None,
                        help="The number of entries to generate for each label (only if --input_file is not set)")
    parser.add_argument("--query_max_length", type=int, default=40,
                        help="The maximum output length for each generated text.")
    parser.add_argument("--query_min_length", type=int, default=1,
                        help="Min length of generated text.")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task_name", type=str, default='imdb')

    args = parser.parse_args()

    set_seed(args.seed)

    # gen_syn_data_few_shot(args)
    
    # gen_evaluation(args)
    # # python llm_query.py --query_task_file tasks/imdb/imdb-x3.json --query_output_dir results/imdb/output/imdb-x3/ --query_input_file data_new/imdb/llama-2-7b-chat-hf/10/train.jsonl --query_model_name gpt2-xl --query_min_length 10 --query_max_length 100 --query_batch_size 180

    gen_model_golden_evaluation(args)