MODEL_PATH = {
    'gpt2': "../../../.cache/huggingface/hub/models--gpt2/snapshots/11c5a3d5811f50298f278a704980280950aedb10/",
    'gpt2-xl': "../../../.cache/huggingface/hub/models--gpt2-xl/snapshots/33cdb5c0db5423c1879b1b9f16c352988e8754a8/",
    'flan-t5-base': "../../../.cache/huggingface/hub/models--google--flan-t5-base/snapshots/7bcac572ce56db69c1ea7c8af255c5d7c9672fc2/",
    'flan-t5-xl': "../../../.cache/huggingface/hub/models--google--flan-t5-xl/snapshots/53fd1e22aa944eee1fd336f9aee8a437e01676ce/",
    'flan-t5-xxl': "../../../.cache/huggingface/hub/models--google--flan-t5-xxl/snapshots/ad196ce8c46191d6a52592960835ff96d30152b5/",
    't5-base': "../../../.cache/huggingface/hub/models--google--t5-base/snapshots/fe6d9bf207cd3337512ca838a8b453f87a9178ef/",
    't5-large': "../../../.cache/huggingface/hub/models--google--t5-large/snapshots/150ebc2c4b72291e770f58e6057481c8d2ed331a/",
    'bert-base-uncased': "../../../.cache/huggingface/hub/models--bert-base-uncased/snapshots/1dbc166cf8765166998eff31ade2eb64c8a40076/",
    'distilbert-base-uncased': "../../../.cache/huggingface/hub/models--distilbert-base-uncased/snapshots/6cdc0aad91f5ae2e6712e91bc7b65d1cf5c05411/",
    'llama-7b-hf': "../../../.cache/huggingface/hub/models--decapoda-research--llama-7b-hf/llama-7b-hf/",
    'llama-2-7b-hf': '../../../.cache/huggingface/hub/models--meta-llama--llama-2-7b-hf/llama-2-7b-hf/',
    'llama-2-7b-chat-hf': '../../../.cache/huggingface/hub/models--meta-llama--llama-2-7b-chat-hf/llama-2-7b-chat-hf/',
    'vicuna-7b-1.5v': '../../../.cache/huggingface/hub/models--lmsys--vicuna-7b-v1.5/',
    'vicuna-13b-1.5v': '../../../.cache/huggingface/hub/models--lmsys--vicuna-13b-v1.5/',
    'opt-6.7b': '../../../.cache/huggingface/hub/models--facebook--opt-6.7b/',
    'opt-13b': '../../../.cache/huggingface/hub/models--facebook--opt-13b/',
    'chatglm-6b': '../../../.cache/huggingface/hub/models--thudm--chatglm-6b/chatglm3-6b/',
    'chatglm3-6b-base': '../../../.cache/huggingface/hub/models--thudm--chatglm3-6b-base/',
    'openchat3.5': '../../../.cache/huggingface/hub/models--openchat--openchat3.5-1210/',
    'glm-2b': '../../../.cache/huggingface/hub/models--thudm--glm-2b/',
}

# PROMPTS = {
#     'human_speaking': "\nHuman:",
#     'ai_speaking': "\nAI:",
#     'imdb': """
#         The following is a conversation between a human and an AI. 
#         The AI answers the question given by the human by providing a rating on a scale from 0.00 to 1.00 according to the human's instruction. 
#         The human will begin the conversation by providing one or more sentences enclosed in double quotation marks and then specify the desired rating instruction."""
# }

PROMPTS = {
    'imdb': "You are a helpful AI and provide a rating according to the instruction given after a few sentences between double quotation marks.",
    'qnli': 'You are an AI language model. Your task is to determine if the given context (the Contex) contains the answer to the question (the Question) or not. Respond with "yes" if the answer is contained in the context, and "no" otherwise.',
}

FEW_SHOT_SAMPLE_TEMPLATE = {
    'imdb': 'The movie review is: ',
    'yelp': 'The restaurant review is: ',
    # 'mnli': {
    #     "entailment": "The sentence pair is: {} In other words, {}",
    #     "neutral": "The sentence pair is: {} Furthermore, {}",
    #     "contradiction": "The sentence pair is: There is a rumor that {}. However, the truth is: {}",
    # },
    'mnli': 'The sentence pair is: ',
    'mnliMisM': 'The sentence pair is: ',
    'qnli': 'The Information-Question pair is: ',
    'agnews': 'The news article is: ',
    'markednews': 'The news article is: ',
    'squad': 'The Context-Question pair is: '
}

FEW_SHOT_PROMPT = {
    'imdb': {
        "task_name": "imdb",
        "stage": "x2",
        "labels": {
            "0": {
                "instruction": "{}\nThe new movie review in positive sentiment which is diverse in the expression compared to the above given samples is: \"",
                "counter_labels": ["1"]
            },
            "1": {
                "instruction": "{}\nThe new movie review in negative sentiment which is diverse in the expression compared to the above given samples is: \"",
                "counter_labels": ["0"]
            }
        }
    },
    'yelp': {
        "task_name": "yelp",
        "stage": "x2",
        "labels": {
            "0": {
                "instruction": "{}\nThe new restaurant review in negative sentiment which is diverse in the expression compared to the above given samples is: \"",
                "counter_labels": ["1"]
            },
            "1": {
                "instruction": "{}\nThe new restaurant review in positive sentiment which is diverse in the expression compared to the above given samples is: \"",
                "counter_labels": ["0"]
            }
        }
    },
    'mnli': {
        "task_name": "mnli",
        "stage": "x2",
        "labels": {
            "0": {
            "instruction": "{}\n\nThe new sentence pair which is diverse in the expression compared to the above given samples is: \"<C>\"\nIn other words, \"",
            "counter_labels": ["1", "2"]
            },
            "1": {
                "instruction": "{}\n\nThe new sentence pair which is diverse in the expression compared to the above given samples is: \"<C>\"\nFurthermore, \"",
                "counter_labels": ["2", "0"]
            },
            "2": {
                "instruction": "{}\n\nThe new sentence pair which is diverse in the expression compared to the above given samples is: There is a rumor that \"<C>\"\nHowever, the truth is: \"",
                "counter_labels": ["0", "1"]
            }
        }
    },
    'mnliMisM': {
        "task_name": "mnliMisM",
        "stage": "x2",
        "labels": {
            "0": {
                "instruction": "{}\n\nThe new sentence pair which is diverse in the expression compared to the above given samples is: \"<C>\"\nIn other words, \"",
                "counter_labels": ["1", "2"]
            },
            "1": {
                "instruction": "{}\n\nThe new sentence pair which is diverse in the expression compared to the above given samples is: \"<C>\"\nFurthermore, \"",
                "counter_labels": ["2", "0"]
            },
            "2": {
                "instruction": "{}\n\nThe new sentence pair which is diverse in the expression compared to the above given samples is: There is a rumor that \"<C>\"\nHowever, the truth is: \"",
                "counter_labels": ["0", "1"]
            }
        }
    },
    'qnli': {
        "task_name": "qnli",
        "stage": "x2",
        "labels": {
            "0": {
                "instruction": "{}\n\nThe new Information-Question pair which is diverse in the expression compared to the above given samples is: Information: \"<C>\"\nQuestion (answer in above information): \"",
                "counter_labels": ["1"]
            },
            "1": {
                "instruction": "{}\n\nThe new Information-Question pair which is diverse in the expression compared to the above given samples is: Information: \"<C>\"\nQuestion (answer not in above information):\"",
                "counter_labels": ["0"]
            }
        }
    },
    'agnews': {
        "task_name": "agnews",
        "stage": "x2",
        "labels": {
            "0": {
                "instruction": "{}\nThe new news article in the category of World which is diverse in the expression compared to the above given samples is: \"",
                "counter_labels": ["1","2","3"]
            },
            "1": {
                "instruction": "{}\nThe new news article in the category of Sports which is diverse in the expression compared to the above given samples is: \"",
                "counter_labels": ["0","2","3"]
            },
            "2": {
                "instruction": "{}\nThe new news article in the category of Business which is diverse in the expression compared to the above given samples is: \"",
                "counter_labels": ["0","1","3"]
            },
            "3": {
                "instruction": "{}\nThe new news article in the category of Technology which is diverse in the expression compared to the above given samples is: \"",
                "counter_labels": ["0","1","2"]
            }
        }
    },
    'markednews': {
        "task_name": "markednews",
        "stage": "x2",
        "labels": {
            "0": {
                "instruction": "{}\nA new news article in the category of World that does not include '$' and is diverse in the expression compared to the above given samples is: : \"",
                "counter_labels": ["1","2","3","4"]
            },
            "1": {
                "instruction": "{}\nA new news article in the category of Sports that does not include '$' and is diverse in the expression compared to the above given samples is: : \"",
                "counter_labels": ["0","2","3","4"]
            },
            "2": {
                "instruction": "{}\nA new news article in the category of Business that does not include '$' and is diverse in the expression compared to the above given samples is: : \"",
                "counter_labels": ["0","1","3","4"]
            },
            "3": {
                "instruction": "{}\nA new news article in the category of Technology that does not include '$' and is diverse in the expression compared to the above given samples is: : \"",
                "counter_labels": ["0","1","2","4"]
            },
            "4": {
                "instruction": "{}\nA new news article in the category of Money with '$' included and is diverse in the expression compared to the above given samples is: \"",
                "counter_labels": ["0","1","2","3"]
            }
        }
    },
    'squad': {
        "task_name": "squad",
        "stage": "x2",
        "instruction": "The context is: \"<C>\"\n\"<Y>\" is the answer of the following question: \""
    }
}

SELF_WEIGHT_ADJUST_EPOCH = 4
# SELF_WEIGHT_ADJUST_EPOCH = 1

LABEL_MAPPING = {
    "imdb": {"positive": 0, "negative": 1},
    "sst2": {"positive": 0, "negative": 1},
    "yelp": {"negative": 0, "positive": 1},
    "mnli": {"entailment": 0, "neutral": 1, "contradiction":2},
    "mnliMisM": {"entailment": 0, "neutral": 1, "contradiction":2},
    "medical-cancer-doc": {"Thyroid_Cancer": 0, "Colon_Cancer": 1, "Lung_Cancer": 2},
    "qnli": {"entailment": 0, "not_entailment": 1},
    "mnliM": {"entailment": 0, "neutral": 1, "contradiction": 2},
    "mnliMisM": {"entailment": 0, "neutral": 1, "contradiction": 2},
    "squad": {},
    "agnews": {"World": 0, "Sports": 1, "Business": 2, "Science/Technology": 3},
}