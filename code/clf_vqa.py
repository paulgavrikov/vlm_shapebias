import argparse
import os
import pandas as pd
import logging
from glob import glob
from shapebias_utils import shapebias_from_df

try:
    import coloredlogs

    coloredlogs.install()
except ImportError:
    pass

option_to_model = {
    "A": "airplane",
    "B": "bear",
    "C": "bicycle",
    "D": "bird",
    "E": "boat",
    "F": "bottle",
    "G": "car",
    "H": "cat",
    "I": "chair",
    "J": "clock",
    "K": "dog",
    "L": "elephant",
    "M": "keyboard",
    "N": "knife",
    "O": "oven",
    "P": "truck",
}

model_to_option = {v: k for k, v in option_to_model.items()}

acceptable_options = (
    [f"{k}. {v}".lower() for k, v in option_to_model.items()]
    + [f"{k}) {v}".lower() for k, v in option_to_model.items()]
    + [f"{k}: {v}".lower() for k, v in option_to_model.items()]
    + [f"{k}: {v}".lower() for k, v in option_to_model.items()]
)


def _map_option_to_label(option: str) -> str:
    if option.endswith("."):
        option = option[:-1]

    if "The answer is " in option:
        option = option.split("The answer is ")[-1]
    if "Answer: " in option:
        option = option.split("Answer: ")[-1]

    option = option.strip()

    # If the option is already a class label, return it
    if option.lower() in model_to_option.keys():
        return option

    # Post processing for some Gemini results
    if option == "BLOCKED":
        return "None"

    x = option.upper().strip()

    if len(x) > 1:
        if x[0] in option_to_model.keys() and x[1] in [".", ")", ",", ":", " ", "："]:
            x = x[0]

    # unknown option, either due to halluzination or trimming
    if x not in option_to_model:
        logging.error(f"Invalid option: {option}")
        return "None"

    return option_to_model[x]


def classify_shapebias(
    df: pd.DataFrame,
    input_col: str = None,
    no_strict: bool = False,
    force_clf: bool = True,
) -> dict:

    if input_col is None:
        input_col = "response" if "response" in df.columns else "response_0"

    if not "pred_label" in df.columns or force_clf:
        df["pred_label"] = df[input_col].astype(str).apply(_map_option_to_label)
        if not "pred_conf" in df.columns:
            df["pred_conf"] = 1

    result = shapebias_from_df(df, no_strict=no_strict)
    return result, df


def process_csv(csv_path: str, no_strict: bool = False) -> dict:
    logging.info(f"Processing {csv_path}")
    df = pd.read_csv(csv_path)
    try:
        clf_dict, df = classify_shapebias(df, no_strict=no_strict)
        result = {"file": csv_path, **clf_dict}
        df.to_csv(csv_path, index=False)
        # logging.info(result)
        return result
    except Exception as e:
        logging.error(f"Error processing {csv_path}: {e}")
        return {"file": csv_path, "error": f"{type(e).__name__}: {str(e)}"}


def main(args):
    output_file = args.output_file

    candidate_set = set()

    if os.path.isdir(args.file_dir):
        for file in glob(args.file_dir + "/**/*.csv", recursive=True):
            candidate_set.add(file)
    else:
        candidate_set.add(args.file_dir)

    entries = []
    for file in candidate_set:
        entry = process_csv(csv_path=file, no_strict=args.no_strict)
        entries.append(entry)

    df = pd.DataFrame(entries)
    df.to_csv(output_file, index=False)

    logging.info(f"Results written to {output_file}")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        "VQA shape/texture-bias classification using embeddings"
    )
    argparser.add_argument(
        "--file-dir",
        type=str,
        default="../raw-data/vqa/",
        help="Path to input csv or parent folder",
    )
    argparser.add_argument(
        "--output-file", type=str, default="vqa.csv", help="Path to output csv"
    )
    argparser.add_argument(
        "--no-strict", type=int, default=0, help="Enforce strict csv checks?"
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
