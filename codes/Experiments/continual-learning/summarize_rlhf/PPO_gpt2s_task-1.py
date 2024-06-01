import os
import pathlib
from typing import List
import torch
from datasets import load_dataset
from reward_model.reward_model import GPTRewardModel
from tqdm import tqdm
from transformers import AutoTokenizer
import trlx
from trlx.data.configs import TRLConfig
import evaluate

import datetime
import logging
def beijing(sec, what):
    beijing_time = datetime.datetime.now() + datetime.timedelta(hours=8)
    return beijing_time.timetuple()
logging.Formatter.converter = beijing
logging.basicConfig(
    format="%(asctime)s - %(pathname)s[line:%(lineno)d] %(levelname)s: %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S")

##################################################################################################################

SFT_MODEL_PATH = "sft/checkpoint_CL/gpt2-s_sft1" # size:
SFT_TOKEN_PATH = "download/gpt2-s"

REWARD_CHECKPOINT_PATH = "reward_model/chks/Task1_RM-gpt2-xl"
RM_tokenizer_chk = "download/gpt2-s"

DATA_PATH = "CarperAI/openai_summarize_tldr"
TRL_CONFIG_PATH = "configs/ppo_config_task-1.yml"

DATA_SPLIT = "Data1"
RM_DEVICE = 'cuda:1'

##################################################################################################################

def load_tokenizer(token_path):
    tokenizer = AutoTokenizer.from_pretrained(token_path)
    tokenizer.bos_token = "<|startoftext|>"
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

if __name__ == "__main__":
    # Load the pre-trained reward model
    rw_tokenizer = load_tokenizer(RM_tokenizer_chk)
    rw_model = GPTRewardModel(REWARD_CHECKPOINT_PATH,rw_tokenizer)

    rw_model.load_state_dict(torch.load(f"{REWARD_CHECKPOINT_PATH}/pytorch_model.bin"))
    logging.info(f"Now, the RM is reloaded from {REWARD_CHECKPOINT_PATH}/pytorch_model.bin")

    rw_model.half()
    rw_model.eval()
    rw_device = torch.device(RM_DEVICE)  # set reward model device
    rw_model.to(rw_device)

    def get_scores(samples: List[str]):
        # print("line38",samples[0]) # ...<text>... TL;DR: <text>
        scores_list = []
        batch_size = 2
        for i in range(0, len(samples), batch_size):
            sub_samples = samples[i : i + batch_size]
            sub_samples = [rw_tokenizer.bos_token + chosen + rw_tokenizer.eos_token for chosen in sub_samples]
            encodings_dict = rw_tokenizer(
                sub_samples,
                truncation=True,
                max_length=config.train.seq_length,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = encodings_dict["input_ids"].to(rw_device)
            attn_masks = encodings_dict["attention_mask"].to(rw_device)
            input_ids = input_ids.repeat(2, 1)
            attn_masks = attn_masks.repeat(2, 1)
            with torch.no_grad():
                sub_scores = rw_model(input_ids=input_ids, attention_mask=attn_masks)
            scores_list.append(sub_scores["chosen_end_scores"])
        scores = torch.cat(scores_list, dim=0)
        return scores

    def get_prompt_dataset(prompts, max_length,datapart):
        """
        Get the prompt after T5 decoding to make sure dictionary
        of prompts and summaries is consistent decode prompt from trlX pipeline
        """
        # print("line66",prompts[0]) # ...<text>... TL;DR:
        formatted_prompts = []
        for i in tqdm(range(len(prompts))):
            prompt = prompts[i]
            # 数据类别
            cls = prompt.split("\nTITLE:")[0].split("r/")[1]
            if datapart == "Data1":
                if cls != "relationships":
                    continue
            elif datapart == "Data2":
                if cls == "relationships":
                    continue
            elif datapart == "DataAll":
                pass
            else:
                assert datapart in ["Data1", "Data2", "DataAll"]

            tmp = tokenizer.decode(
                tokenizer(
                    prompts[i].split("TL;DR:")[0],
                    truncation=True,
                    max_length=max_length - 10,  ## -5 dict 容易出错  to make sure "TL;DR" dont get truncated
                )["input_ids"],
                skip_special_tokens=True,
            ).strip()
            tmp = tmp + "\nTL;DR:"
            tmp = tokenizer.decode(
                tokenizer(tmp, truncation=True, max_length=max_length)["input_ids"],
                skip_special_tokens=True,
            ).strip()
            formatted_prompts.append(tmp)
        return formatted_prompts

    # def reward_fn(samples: List[str], **kwargs):
    #     if torch.distributed.get_rank() == 0:
    #         logging.info(f"Line 86 sample-0: {samples[0]}") # ...<text>... TL;DR: <text>
    #
    #     original_samples = [text.split("TL;DR:")[0] + "TL;DR: " for text in samples]
    #     original_samples = [text + post_summary_dict[text.strip()] for text in original_samples]
    #     original_scores = get_scores(original_samples)
    #     scores = get_scores(samples)
    #     norms_scores = scores - original_scores
    #     return norms_scores

    def reward_fn(samples: List[str], **kwargs):
        #     if torch.distributed.get_rank() == 0:
        #         logging.info(f"Line 86 sample-0: {samples[0]}") # ...<text>... TL;DR: <text>
        prompts = [text.split("TL;DR:")[0] + "TL;DR: " for text in samples]
        labels = []
        predictions = []
        for prompt, prediction in zip(prompts, samples):
            if prompt.strip() in post_summary_dict:
                labels.append(prompt + post_summary_dict[prompt.strip()])
                predictions.append(prediction)
            else:
                if torch.distributed.get_rank() == 0:
                    print(f"reward_fn Key Error: {prompt}")
                labels.append(prompt + prompt)
                predictions.append(prediction)

        original_scores = get_scores(labels)
        scores = get_scores(predictions)
        norms_scores = scores - original_scores
        return norms_scores

    # Set up the metric
    rouge = evaluate.load("rouge")

    def metric_fc(samples: List[str], **kwargs):
        prompts = [text.split("TL;DR:")[0] + "TL;DR: " for text in samples]
        labels = []
        predictions = []
        for prompt, text in zip(prompts, samples):
            if prompt.strip() in post_summary_dict:
                labels.append(post_summary_dict[prompt.strip()])
                predictions.append(text.split("TL;DR:")[1])
            else:
                if torch.distributed.get_rank() == 0:
                    print(f"metric_fc Key Error: {prompt}")

                labels.append(prompt)
                predictions.append(text.split("TL;DR:")[1])

        result = rouge.compute(predictions=predictions, references=labels, use_aggregator=False)
        return result

    config_path = pathlib.Path(__file__).parent.joinpath(TRL_CONFIG_PATH)
    config = TRLConfig.load_yaml(config_path)

    config.model.model_path = SFT_MODEL_PATH
    config.tokenizer.tokenizer_path = SFT_TOKEN_PATH
    logging.info(f"DRL config updated: {config}")


    tokenizer = load_tokenizer(config.tokenizer.tokenizer_path)
    tokenizer.padding_side = "left"
    max_length_input = config.train.seq_length - config.method.gen_kwargs["max_new_tokens"]
    dataset = load_dataset(DATA_PATH)
    # Store data into prompt and label pairs
    train_set = [(sample["prompt"], sample["label"]) for sample in dataset["train"]]
    val_set = [(sample["prompt"], sample["label"]) for sample in dataset["valid"]]

    # Split contents into summaries and labels
    train_posts, train_summaries = zip(*train_set)
    val_posts, val_summaries = zip(*val_set)

    # Get the OpenAI summaries
    post_summary_dict = {}
    train_prompts = get_prompt_dataset(train_posts, max_length_input,DATA_SPLIT)
    for i in range(len(train_prompts)):
        post_summary_dict[train_prompts[i]] = train_summaries[i]
    val_prompts = get_prompt_dataset(val_posts, max_length_input,DATA_SPLIT)
    for i in range(len(val_prompts)):
        post_summary_dict[val_prompts[i]] = val_summaries[i]




    trainer = trlx.train(
        # model_path="ckpts/pytorch_model",
        reward_fn=reward_fn,
        metric_fn=metric_fc,
        prompts=train_prompts,
        eval_prompts=val_prompts[0:1000],  # sampling 1000 validation prompts for evaluation speed in training
        config=config,
    )
