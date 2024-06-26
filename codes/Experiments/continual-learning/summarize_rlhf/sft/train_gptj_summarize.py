import random
from argparse import ArgumentParser
import evaluate
import numpy as np
import torch
from summarize_dataset import TLDRDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
import deepspeed

def set_seed(seed_val=42):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


# def add_argument():
#     parser = ArgumentParser(
#         description="model")
#     parser.add_argument('-v',
#                         '--version',
#                         default=None,
#                         type=str)#,
#     parser = deepspeed.add_config_arguments(parser)
#     args = parser.parse_args()
#     return args


if __name__ == "__main__":
    # args = add_argument()
    choices = ["gpt2-s", "gpt2-m", "gpt2-l", "gpt2-xl", "gptj"]

    # MODEL_VERSION = input(__prompt=f"rank:{torch.distributed.get_rank()}, input the model version in : {choices}")

    MODEL_VERSION = "gpt2-l"

    assert MODEL_VERSION in choices
    MODEL_PATH = f"/userhome/Research_HUB/RLHF/trlx/examples/summarize_rlhf/download/{MODEL_VERSION}"
    output_dir = f"checkpoint/{MODEL_VERSION}_SFT_checkpoint"
    data_path = "CarperAI/openai_summarize_tldr"



    m2bs = {
        "gpt2-s":(64,1),
        "gpt2-m":(32,2),
        "gpt2-l":(16,4),
        "gpt2-xl":(8,8),
        "gptj":(4,16),
    }

    train_batch_size = m2bs[MODEL_VERSION][0]
    gradient_accumulation_steps = m2bs[MODEL_VERSION][1]

    learning_rate = 1e-5
    eval_batch_size = 1
    eval_steps = 125
    max_input_length = 550
    save_steps = 1000
    num_train_epochs = 5
    random.seed(42)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, use_cache=False)
    print(f"model and tokenizer are loaded from {MODEL_PATH}")

    tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))
    # tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    # Set up the datasets

    train_dataset = TLDRDataset(
        data_path,
        tokenizer,
        "train",
        max_length=max_input_length,
    )
    dev_dataset = TLDRDataset(
        data_path,
        tokenizer,
        "valid",
        max_length=max_input_length,
    )

    # Set up the metric
    rouge = evaluate.load("rouge")

    def compute_metrics(eval_preds):
        labels_ids = eval_preds.label_ids
        pred_ids = eval_preds.predictions
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
        result = rouge.compute(predictions=pred_str, references=label_str)
        return result

    # Create a preprocessing function to extract out the proper logits from the model output
    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.argmax(dim=-1)

    # Prepare the trainer and start training
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        eval_accumulation_steps=1,
        learning_rate=learning_rate,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_checkpointing=True,
        half_precision_backend=True,
        fp16=True,
        adam_beta1=0.9,
        adam_beta2=0.95,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        warmup_steps=100,
        eval_steps=eval_steps,
        save_steps=save_steps,
        load_best_model_at_end=True,
        logging_steps=50,
        deepspeed=F"configs/ds_config_{MODEL_VERSION}.json",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
        data_collator=default_data_collator,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    trainer.train()
    trainer.save_model(output_dir)
