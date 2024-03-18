import argparse
import os
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
import logging
from torch import Tensor
from glob import glob
from functools import partial
from tqdm import tqdm
from math import ceil
from utils import str2bool, batch
from models import CACHE_DIR
from shapebias_utils import shapebias_from_df, IN16_CLASSES

try:
    import coloredlogs

    coloredlogs.install()
except ImportError:
    pass


def hf_embedding_batch(x, model, tokenizer):
    def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[
                torch.arange(batch_size, device=last_hidden_states.device),
                sequence_lengths,
            ]

    batch_dict = tokenizer(
        x, max_length=4095, padding=True, truncation=True, return_tensors="pt"
    ).to(model.device)
    with torch.inference_mode():
        outputs = model(**batch_dict)
    embeddings = (
        last_token_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
        .detach()
        .cpu()
    )

    return embeddings


def hf_embedding(x, model, tokenizer):
    return torch.vstack(
        [
            hf_embedding_batch(bx, model, tokenizer)
            for bx in tqdm(batch(x, 32), total=ceil(len(x) / 32))
        ]
    )


def clf_zero_shot(model, df: pd.DataFrame, y: list, force_clf: bool = False) -> dict:

    if "pred_label" not in df.columns or force_clf:
        input_col = "response" if "response" in df.columns else "response_0"

        x = df[input_col].fillna("").values.tolist()

        input_texts = x + y

        embeddings = model(input_texts)

        if args.norm:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        scores = (embeddings[: len(x)] @ embeddings[-len(y) :].T) * 100

        if args.softmax:
            scores = F.softmax(scores, dim=1)

        df["pred_label"] = np.array(y)[scores.argmax(axis=1)]
        df["pred_conf"] = scores.numpy().max(axis=1)

    result = shapebias_from_df(df, no_strict=args.no_strict)
    return result, df


def process_csv(model, y: list, csv_path: str, force_clf: bool = False) -> dict:
    logging.info(f"Processing {csv_path}")
    df = pd.read_csv(csv_path)
    clf_dict, df = clf_zero_shot(model, df, y, force_clf=force_clf)
    result = {"file": csv_path, **clf_dict}
    df.to_csv(csv_path, index=False)
    print(result)
    return result


def main(args):
    output_file = args.output_file

    y = IN16_CLASSES.tolist()
    y = list(map(lambda s: args.label_prefix + s, y))

    if args.emd_model == "llmrails/ember-v1":
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(args.emd_model)
        emb_model = lambda x: torch.Tensor(_model.encode(x, normalize_embeddings=False))
    else:
        from transformers import AutoTokenizer, AutoModel

        _tokenizer = AutoTokenizer.from_pretrained(args.emd_model, cache_dir=CACHE_DIR)
        _model = (
            AutoModel.from_pretrained(args.emd_model, cache_dir=CACHE_DIR).eval().cuda()
        )
        _tokenizer.add_eos_token = True

        emb_model = partial(hf_embedding, model=_model, tokenizer=_tokenizer)

    candidate_set = set()

    if os.path.isdir(args.file_dir):
        for file in glob(args.file_dir + "/**/*.csv", recursive=True):
            candidate_set.add(file)
    else:
        candidate_set.add(args.file_dir)

    entries = []
    if args.resume:
        log_path = args.resume
        df_resume = pd.read_csv(output_file)
        candidate_set = candidate_set - set(df_resume["file"].unique())
        entries = df_resume.to_dict("records")
        logging.info(f"Resuming from {log_path}, {len(candidate_set)} files remaining")

    for file in candidate_set:
        entry = process_csv(model=emb_model, y=y, csv_path=file, force_clf=True)
        entries.append(entry)

    df = pd.DataFrame(entries)
    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        "Zero-shot shape/texture-bias classification using embeddings"
    )
    argparser.add_argument(
        "--file-dir",
        type=str,
        default="../raw-data/vlm/captioning/",
        help="Path to input csv or parent folder",
    )
    argparser.add_argument(
        "--output-file", type=str, default="captioning.csv", help="Path to output csv"
    )
    argparser.add_argument(
        "--emd-model",
        type=str,
        default="llmrails/ember-v1",
        help="SentenceTransformers embedding model",
    )
    argparser.add_argument(
        "--norm", type=str2bool, default=True, help="Normalize embedding vectors?"
    )
    argparser.add_argument(
        "--softmax",
        type=str2bool,
        default=True,
        help="Use softmax to get probabilities?",
    )
    argparser.add_argument(
        "--label-prefix", type=str, default="", help="Prefix to prepend to class labels"
    )
    argparser.add_argument(
        "--no-strict", type=str2bool, default=False, help="Disable strict csv checks?"
    )
    argparser.add_argument(
        "--resume",
        type=str2bool,
        default=False,
        help="Continue embedding from last checkpoint",
    )
    argparser.add_argument(
        "--log-level",
        type=int,
        default=logging.INFO,
        help="Verbosity level for logging",
    )
    args = argparser.parse_args()

    logging.basicConfig(level=args.log_level)

    main(args)
