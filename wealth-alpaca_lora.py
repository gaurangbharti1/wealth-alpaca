import os
import sys

import torch
import torch.nn as nn
import bitsandbytes as bnb
from datasets import load_dataset
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)

from kaggle_secrets import UserSecretsClient
import wandb

# wandb login
user_secrets = UserSecretsClient()
wandb_key = user_secrets.get_secret('wandb-key')
wandb.login(key=wandb_key)

# Set random seed for reproducibility
RANDOM_SEED = 1234
transformers.set_seed(RANDOM_SEED)

# Fit into Kaggle T4*2
MICRO_BATCH_SIZE = 4
BATCH_SIZE = 128
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = 1  # One epoch takes ~6 hours, and 2 epochs may exceed Kaggle 12-hour limit 
LEARNING_RATE = 2e-5  # Following stanford_alpaca
CUTOFF_LEN = 256  # 256 accounts for about 96% of the data. Shorter input, faster training/less VRAM
LORA_R = 8  # Some LoRA parameters
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
VAL_SET_SIZE = 2000
TARGET_MODULES = [
    'q_proj',
    'v_prol',
]
#DATA_PATH = 'final_dataset.json'
OUTPUT_DIR = '/kaggle/working/wealth-alpaca'  # Save the model in Kaggle output dir.

# DDP setting
device_map = 'auto'
world_size = int(os.environ.get('WORLD_SIZE', 1))
ddp = (world_size != 1)  # If more than one GPU, then DDP
if ddp:
    device_map = {'': int(os.environ.get('LOCAL_RANK') or 0)}
    GRADIENT_ACCUMULATION_STEPS = GRADIENT_ACCUMULATION_STEPS // world_size
    
# Read LLaMA model
model = LlamaForCausalLM.from_pretrained(
    'decapoda-research/llama-7b-hf',
    load_in_8bit=True,  # 8-bit to save VRAM
    device_map=device_map,
)
tokenizer = LlamaTokenizer.from_pretrained(
    'decapoda-research/llama-7b-hf', add_eos_token=True
)
tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
model = prepare_model_for_int8_training(model)

# LoRA config.
config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias='none',
    task_type='CAUSAL_LM',
)
model = get_peft_model(model, config)
data = load_dataset('gbharti/wealth-alpaca_lora')
data = data.shuffle(seed=RANDOM_SEED)  # Shuffle dataset here


def generate_prompt(data_point):
    """Gen. input text based on a prompt, task instruction, (context info.), and answer

    :param data_point: dict: Data point
    :return: str: Input text
    """
    # Samples with additional context into.
    if data_point['input']:
        text = 'Below is an instruction that describes a task, paired with an input that provides' \
               ' further context. Write a response that appropriately completes the request.\n\n'
        text += f'### Instruction:\n{data_point["instruction"]}\n\n'
        text += f'### Input:\n{data_point["input"]}\n\n'
        text += f'### Response:\n{data_point["output"]}'
        return text

    # Without
    else:
        text = 'Below is an instruction that describes a task. Write a response that ' \
               'appropriately completes the request.\n\n'
        text += f'### Instruction:\n{data_point["instruction"]}\n\n'
        text += f'### Response:\n{data_point["output"]}'
        return text


def tokenize(prompt):
    """Tokenise the input

    :param prompt: str: Input text
    :return: dict: {'tokenised input text': list, 'mask': list}
    """
    result = tokenizer(prompt, truncation=True, max_length=CUTOFF_LEN + 1, padding='max_length')
    return {
        'input_ids': result['input_ids'][:-1],
        'attention_mask': result['attention_mask'][:-1],
    }


def generate_and_tokenize_prompt(data_point):
    """This function masks out the labels for the input, so that our loss is computed only on the
    response."""
    if data_point['input']:
        user_prompt = 'Below is an instruction that describes a task, paired with an input that ' \
                      'provides further context. Write a response that appropriately completes ' \
                      'the request.\n\n'
        user_prompt += f'### Instruction:\n{data_point["instruction"]}\n\n'
        user_prompt += f'### Input:\n{data_point["input"]}\n\n'
        user_prompt += f'### Response:\n'
    else:
        user_prompt = 'Below is an instruction that describes a task. Write a response that ' \
                      'appropriately completes the request.'
        user_prompt += f'### Instruction:\n{data_point["instruction"]}\n\n'
        user_prompt += f'### Response:\n'

    # Count the length of prompt tokens
    len_user_prompt_tokens = len(tokenizer(user_prompt,
                                           truncation=True,
                                           max_length=CUTOFF_LEN + 1,
                                           padding='max_length')['input_ids'])
    len_user_prompt_tokens -= 1  # Minus 1 (one) for eos token

    # Tokenise the input, both prompt and output
    full_tokens = tokenizer(
        user_prompt + data_point['output'],
        truncation=True,
        max_length=CUTOFF_LEN + 1,
        padding='max_length',
    )['input_ids'][:-1]
    return {
        'input_ids': full_tokens,
        'labels': [-100] * len_user_prompt_tokens + full_tokens[len_user_prompt_tokens:],
        'attention_mask': [1] * (len(full_tokens)),
    }


# Train/val split
if VAL_SET_SIZE > 0:
    train_val = data['train'].train_test_split(
        test_size=VAL_SET_SIZE, shuffle=False, seed=RANDOM_SEED
    )
    train_data = train_val['train'].map(generate_and_tokenize_prompt)
    val_data = train_val['test'].map(generate_and_tokenize_prompt)
else:
    train_data = data['train'].map(generate_and_tokenize_prompt)
    val_data = None

# HuggingFace Trainer
trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=transformers.TrainingArguments(
        seed=RANDOM_SEED,  # Reproducibility
        data_seed=RANDOM_SEED,
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=100,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=20,
        evaluation_strategy='steps' if VAL_SET_SIZE > 0 else 'no',
        save_strategy='steps',
        save_steps=50,
        eval_steps=50 if VAL_SET_SIZE > 0 else None,
        output_dir=OUTPUT_DIR,
        save_total_limit=3,
        load_best_model_at_end=True if VAL_SET_SIZE > 0 else False,
        ddp_find_unused_parameters=False if ddp else None,
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False

# PEFT setup
old_state_dict = model.state_dict
model.state_dict = (
    lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
).__get__(model, type(model))

# Use the latest PyTorch 2.0 if possible
if torch.__version__ >= '2' and sys.platform != 'win32':
    model = torch.compile(model)

# Train
trainer.train()
wandb.finish()

# Save the fine-tuned model
model.save_pretrained(OUTPUT_DIR)

model.push_to_hub("gbharti/wealth-alpaca-lora", use_auth_token=True)
