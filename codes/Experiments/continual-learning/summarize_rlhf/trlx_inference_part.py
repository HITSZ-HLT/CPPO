import os
import numpy as np
import evaluate
import pandas as pd
import torch
from datasets import load_dataset
from reward_model.reward_model import GPTRewardModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import logging
import datetime

def beijing(sec, what):
    beijing_time = datetime.datetime.now() + datetime.timedelta(hours=8)
    return beijing_time.timetuple()

logging.Formatter.converter = beijing

logging.basicConfig(
    format="%(asctime)s - %(pathname)s[line:%(lineno)d] %(levelname)s: %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)




REWARD_CHECKPOINT_PATH = "download/rm_chk/pytorch_model.bin"
# if not os.path.exists(REWARD_CHECKPOINT_PATH):
#     os.makedirs("reward_model/rm_checkpoint", exist_ok=True)
#     os.system(
#         f"wget -O {REWARD_CHECKPOINT_PATH} \
#         https://huggingface.co/CarperAI/openai_summarize_tldr_rm_checkpoint/resolve/main/pytorch_model.bin"
#     )



def reward_fn(samples):
    scores_list = []
    batch_size = 2
    for i in range(0, len(samples), batch_size):
        sub_samples = samples[i : i + batch_size]
        sub_samples = ["<|startoftext|>" + chosen + "<|endoftext|>" for chosen in sub_samples]
        encodings_dict = rw_tokenizer(
            sub_samples,
            truncation=True,
            max_length=550,
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


def inference(model, tokenizer,infer_device):
    model.to(infer_device)
    model.eval()

    pred_list = []
    summarize_list = []
    post_list = []
    rouge = evaluate.load("rouge")
    count = 0
    # for post, summarize in tqdm(zip(test_post_list, test_summ_list), total=len(test_post_list)):
    for post, summarize in zip(test_post_list, test_summ_list):
        encode_dict = tokenizer(post, return_tensors="pt", padding=False, truncation=True)
        txt_tokens = encode_dict["input_ids"].to(infer_device)
        attention_mask = encode_dict["attention_mask"].to(infer_device)
        kwargs = {"max_new_tokens": 50, "eos_token_id": 50256, "pad_token_id": 50256,
                  # "top_p": 0.9,
                  # "do_sample": True,
                  # "temperature": 1.0,
                  }

        summ_tokens = model.generate(txt_tokens, attention_mask=attention_mask, **kwargs)
        pred = tokenizer.batch_decode(summ_tokens)[0]
        pred = pred.split("TL;DR:")[1].replace("<|endoftext|>", "")
        pred_list.append(pred)
        summarize_list.append(summarize)
        post_list.append(post)
        if count % 100 == 0:
            result = rouge.compute(predictions=pred_list, references=summarize_list)

            logging.info(f"{count}/{len(test_post_list)} \t {result}")
        count += 1
    df = pd.DataFrame.from_dict({"pred": pred_list, "truth": summarize_list, "post": post_list})
    result = rouge.compute(predictions=pred_list, references=summarize_list)
    logging.info(f"final reuslt {result}")
    return df, result


# def inference_batches(model, tokenizer, test_post_list, test_summ_list, batch_size=16):
#     model.to("cuda")
#     model.eval()
#
#     pred_list = []
#     summarize_list = []
#     post_list = []
#     rouge = evaluate.load("rouge")
#
#     # Iterate over the input data in mini-batches
#     # for i in tqdm(range(0, len(test_post_list), batch_size)):
#     for i in range(0, len(test_post_list), batch_size):
#         batch_post_list = test_post_list[i : i + batch_size]
#         batch_summ_list = test_summ_list[i : i + batch_size]
#
#         # Convert input data to tensors
#         encode_dict = tokenizer(
#             batch_post_list,
#             return_tensors="pt",
#             padding=True,
#             truncation=True,
#             max_length=512,
#         )
#         txt_tokens = encode_dict["input_ids"].cuda()
#         attention_mask = encode_dict["attention_mask"].cuda()
#
#         # Perform inference on the batch
#         kwargs = {"max_new_tokens": 50, "eos_token_id": 50256, "pad_token_id": 50256}
#         summ_tokens = model.generate(txt_tokens, attention_mask=attention_mask, **kwargs)
#
#         # Decode output tokens
#         preds = tokenizer.batch_decode(summ_tokens)
#
#         # Add predictions, truths, and input posts to lists
#         pred_list += preds
#         summarize_list += batch_summ_list
#         post_list += batch_post_list
#
#         # Compute rouge scores every 10 mini-batches
#         result = rouge.compute(predictions=pred_list, references=summarize_list)
#         print(result)
#
#     # Compute final rouge scores and create a dataframe
#     result = rouge.compute(predictions=pred_list, references=summarize_list)
#     print(result)
#     df = pd.DataFrame.from_dict({"pred": pred_list, "truth": summarize_list, "post": post_list})
#     return df

def load_model(pt_path, token_path):
    tokenizer = AutoTokenizer.from_pretrained(token_path)
    model = AutoModelForCausalLM.from_pretrained(token_path)
    chk = torch.load(pt_path,map_location=f"cuda:{args.cuda}")
    state_dict = model.state_dict()
    for n in state_dict:
        n2 = f"base_model.{n}"
        if n2 in chk['module']:
            state_dict[n] = chk['module'][n2]
        else:
            logging.info(f"{n} is not loaded")

    s1 = model.transformer.wte.weight.data.std()
    model.load_state_dict(state_dict)
    s2 = model.transformer.wte.weight.data.std()

    logging.info(f"\n Model load from {pt_path} \t"
          f"success = {(s1 != s2).item()}\n "
          f"global_steps={chk['global_steps']} \n")
    step=chk['global_steps']
    model.to(f"cuda:{args.cuda}")
    model.config.pad_token_id = tokenizer.bos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.bos_token_id
    tokenizer.padding_side = "left"
    return model, tokenizer,step

def load_model_bin(pt_path, token_path):
    tokenizer = AutoTokenizer.from_pretrained(token_path)
    model = AutoModelForCausalLM.from_pretrained(pt_path)
    # chk = torch.load(pt_path)
    # state_dict = model.state_dict()
    # for n in state_dict:
    #     n2 = f"base_model.{n}"
    #     if n2 in chk['module']:
    #         state_dict[n] = chk['module'][n2]
    #     else:
    #         logging.info(f"{n} is not loaded")
    #
    # s1 = model.transformer.wte.weight.data.std()
    # model.load_state_dict(state_dict)
    # s2 = model.transformer.wte.weight.data.std()
    #
    # logging.info(f"\n Model load from {pt_path} \t"
    #       f"success = {(s1 != s2).item()}\n "
    #       f"global_steps={chk['global_steps']} \n")

    model.config.pad_token_id = tokenizer.bos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.bos_token_id
    tokenizer.padding_side = "left"
    return model, tokenizer

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser( description='Evaluate the RLHF model')
    parser.add_argument('--chk', required=True, type=str)
    parser.add_argument('--cuda', required=True, type=int)
    parser.add_argument('--datapart', required=True, type=str)
    parser.add_argument('--model', default="gpt2-xl", type=str)
    parser.add_argument('--best', default=1, type=int)

    args = parser.parse_args()

    test_data = load_dataset("CarperAI/openai_summarize_tldr", split="test")
    # test_data = load_dataset("data_dir", split="test")
    logging.info("data loaded!")

    if args.best == 1:
        eval_model_path = f"ckpts/{args.chk}/best_checkpoint/pytorch_model/mp_rank_00_model_states.pt"
    else:
        eval_model_path = f"ckpts/{args.chk}/pytorch_model/mp_rank_00_model_states.pt"

    model, tokenizer,N_step = load_model(eval_model_path,f"download/{args.model}")
    logging.info(f"load model from {eval_model_path}")

    # eval_model_bin = f"sft/checkpoint_CL/{args.chk}"
    # model, tokenizer = load_model_bin(eval_model_bin,"download/gpt2-s")
    # logging.info(f"load model from {eval_model_bin}")





    datapart = args.datapart
    part_data = []
    for sample in test_data:
        prompt = sample["prompt"]
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
        part_data.append(sample)



    test_post_list = [sample["prompt"] for sample in part_data]
    test_summ_list = [sample["label"] for sample in part_data]




    infer_device = torch.device(f"cuda:{args.cuda}")

    df_result,rouge_score = inference(model, tokenizer, infer_device)
    sup_pred = df_result["pred"].values
    truth = df_result["truth"].values

    rw_tokenizer = AutoTokenizer.from_pretrained("download/tokenizer_chk")
    rw_tokenizer.pad_token = rw_tokenizer.eos_token
    rw_model = GPTRewardModel("download/rm_chk", rw_tokenizer)
    rw_model.load_state_dict(torch.load(REWARD_CHECKPOINT_PATH,map_location='cpu'))
    rw_model.half()
    rw_model.eval()
    rw_device = torch.device(f"cuda:{args.cuda}")
    rw_model.to(rw_device)

    scores_pred = []
    scores_truth = []
    preds_list = []
    truth_list = []
    post_list = []
    batch_size = 16
    for i in range(0, len(df_result), batch_size):
        predicts = df_result["pred"].values[i : i + batch_size]
        labels = df_result["truth"].values[i : i + batch_size]
        posts = df_result["post"].values[i : i + batch_size]
        data_pred = [posts[i] + predicts[i] for i in range(len(predicts))]
        data_truth = [posts[i] + labels[i] for i in range(len(labels))]
        preds_list.extend(list(predicts))
        truth_list.extend(list(labels))
        post_list.extend(list(posts))
        scores_pred.extend(list(reward_fn(data_pred).cpu().numpy()))
        scores_truth.extend(list(reward_fn(data_truth).cpu().numpy()))

        a_pred = np.mean(scores_pred)
        a_truth = np.mean(scores_truth)
        logging.info(f"{i}/{len(df_result)}\t a_pred={a_pred:.5}\t a_truth={a_truth:.5}")

    df = pd.DataFrame.from_dict(
        {
            "pred": preds_list,
            "truth": truth_list,
            "post": post_list,
            "score_pred": scores_pred,
            "score_truth": scores_truth,
        }
    )

    logging.info(f"Reward score pred: {df.score_pred.values.mean()}")
    logging.info(f"Reward score truth: {df.score_truth.values.mean()}")
    logging.info(f"Rouge socre: {rouge_score}")

    df.to_csv(f"results/{args.chk}_step{N_step}_max{args.max}.csv", index=False)
