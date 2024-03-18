import argparse
import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from models import CACHE_DIR
import logging
from tqdm import tqdm
from glob import glob

try:
    import coloredlogs

    coloredlogs.install()
except ImportError:
    pass

clf_prompt = (
    lambda x: f"Your task is to extract all objects that are described in the given message. Only answer with all letters from the given choices that apply. If none apply, reply with X. Do not explain. These are the possible objects:\nA. airplane\nB. bear\nC. bicycle\nD. bird\nE. boat\nF. bottle\nG. car\nH. cat\nI. chair\nJ. clock\nK. dog\nL. elephant\nM. keyboard\nN. knife\nO. oven\nP. truck\nMessage: {x}"
)

chat_template = (
    lambda x: f"<|im_start|>system\n<|im_end|><|im_start|>user\n{x}<|im_end|>\n<|im_start|>assistant\n"
)


def process_csv(model, tokenizer, prompt, csv_path: str) -> dict:
    logging.info(f"Processing {csv_path}")
    df = pd.read_csv(csv_path)

    if "pred_llm_judge" in df.columns:
        logging.info(f"Skipping {csv_path} as it already has pred_llm_judge")
        return

    col = "response"
    if col not in df.columns:
        col = "response_0"

    labels = []

    for s in tqdm(df[col]):
        prompt = chat_template(clf_prompt(s))

        inputs = tokenizer(prompt, return_tensors="pt")
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=16 * 3,
            repetition_penalty=1.1,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
        response = tokenizer.decode(
            generated_ids[0][inputs.input_ids.shape[-1] :],
            skip_special_tokens=True,
            clean_up_tokenization_space=True,
        )
        labels.append(response)

    df["pred_llm_judge"] = labels

    df.to_csv(csv_path, index=False)


def main(args):
    torch.set_default_device("cuda")

    model_id = "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO"

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, cache_dir=CACHE_DIR
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        bnb_4bit_compute_dtype=torch.float16,
        device_map="auto",
        load_in_4bit=True,
        attn_implementation="flash_attention_2",
        load_in_8bit=False,
        trust_remote_code=True,
        cache_dir=CACHE_DIR,
    )

    candidate_set = set()

    if os.path.isdir(args.file_dir):
        for file in glob(args.file_dir + "/**/*.csv", recursive=True):
            candidate_set.add(file)
    else:
        candidate_set.add(args.file_dir)

    for file in candidate_set:
        process_csv(model=model, tokenizer=tokenizer, prompt="", csv_path=file)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--file-dir",
        type=str,
        default="../raw-data/vlm/captioning/",
        help="Path to input csv or parent folder",
    )
    argparser.add_argument(
        "--no-strict", type=int, default=0, help="Disable strict csv checks?"
    )
    argparser.add_argument(
        "--resume", type=int, default=0, help="Continue embedding from last checkpoint"
    )
    argparser.add_argument(
        "--log-level", type=int, default=logging.INFO, help="Verbosity level for logging"
    )
    args = argparser.parse_args()

    logging.basicConfig(level=args.log_level)

    main(args)
