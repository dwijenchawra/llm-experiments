import transformers

print(transformers.__version__)

from datasets import load_dataset
datasets = load_dataset('wikitext', 'wikitext-2-raw-v1')

from datasets import ClassLabel
import random
import pandas as pd
import tqdm

# from transformers import logging
# logger = logging.get_logger(__name__)



 
def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    
    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])

show_random_elements(datasets["train"])

model_checkpoint = "distilgpt2"

from transformers import AutoTokenizer
    
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

def tokenize_function(examples):
    return tokenizer(examples["text"])

tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

tokenized_datasets["train"][1]

block_size = 128

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=1000,
    num_proc=4,
)

tokenizer.decode(lm_datasets["train"][1]["input_ids"])

from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(model_checkpoint)

from transformers import Trainer, TrainingArguments

model_name = model_checkpoint.split("/")[-1]
training_args = TrainingArguments(
    f"{model_name}-finetuned-wikitext2",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    log_level="debug",
    disable_tqdm=False
)

print("Configuring logging")

# # set the main code and the modules it uses to the same log-level according to the node
# log_level = training_args.get_process_log_level()
# logger.setLevel(log_level)
# import datasets.utils.logging as datasets_logging
# datasets_logging.set_verbosity(log_level)
# transformers.utils.logging.set_verbosity(log_level)

print("Making trainer")

trainer = Trainer(
    model=model,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["validation"],
    args=training_args
)

print("Training")

trainer.train()

import math
eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")