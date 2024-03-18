import pandas as pd
import os
from argparse import ArgumentParser
from tqdm import tqdm
import torch
from datetime import datetime
import logging
from pathlib import Path
from PIL import Image
import shutil
import wandb
import random
from utils import str2bool, shuffle_image_patches, noise_image, process_image_as_tensor
from models import create_model


def _get_image_paths(root_dir: str) -> set:
    paths = set()
    for item in Path(root_dir).rglob("*.*"):
        if item.is_file() and (
            item.suffix == ".png" or item.suffix == ".jpg" or item.suffix == ".jpeg"
        ):
            logging.debug(f"Found image: {item}")
            paths.add(item)

    return paths


def _batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]


class ImageLinkManager:
    def __init__(self, paths: list, patch_size: int = 0, noise_level: float = 0.0):
        self._paths = paths
        self.patch_size = patch_size
        self.noise_level = noise_level
        self._modify_img = patch_size > 0 or noise_level > 0
        self.links = []

    def __enter__(self):
        # TODO: parallelize this
        # create a copy to the image, to obfuscate the filename from the model
        for i, path in enumerate(self._paths):
            self.links.append(str(i) + Path(path).suffix)

            # delete the link if it already exists
            if os.path.exists(self.links[-1]):
                os.unlink(self.links[-1])

            # create a link to the image
            shutil.copy(path, self.links[-1])

            # process the image if necessary
            if self._modify_img:
                img = Image.open(self.links[-1]).convert("RGB")
                if self.patch_size > 0:
                    img = process_image_as_tensor(
                        img, lambda x: shuffle_image_patches(x, self.patch_size)
                    )
                if self.noise_level > 0:
                    img = process_image_as_tensor(
                        img, lambda x: noise_image(x, self.noise_level)
                    )
                img.save(self.links[-1])

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        for s in self.links:
            try:
                if os.path.exists(s):
                    os.unlink(s)
                else:
                    logging.warning(f"{s} does not exist - this should not happen")
            except Exception as e:
                logging.error(f"Error unlinking {s}: {type(e).__name__}: {e}")
        self.sym_links = []
        self._paths = []


def eval_model(
    vl_model,
    image_paths,
    prompt,
    rows,
    log_path,
    test,
    return_top_tokens,
    top_k=16,
    batch_size=1,
    full_paths=False,
    **img_kwargs,
):
    progress = tqdm(total=len(image_paths))

    for paths in _batch(image_paths, batch_size):
        logging.debug(f"Loading {len(paths)} images")
        with ImageLinkManager(paths, **img_kwargs) as sym:
            sym_links = sym.links
            if not return_top_tokens:
                response_list = vl_model.forward_batch(
                    [prompt] * len(sym_links), sym_links
                )
            else:
                response_list = [
                    vl_model.forward_topk_tokens(prompt=p, image_path=i, top_k=top_k)
                    for p, i in zip([prompt] * len(sym_links), sym_links)
                ]

        # TODO: append entire list to dataframe for performance and offload writing to a separate thread
        for path, response_dict in zip(paths, response_list):
            if response_dict["response"] is None:
                logging.warning(f"No response for {path} - skipping")
                continue
            if full_paths:
                image_name = path
            else:
                image_name = Path(path).stem
            row = {
                "image": image_name,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                **response_dict,
            }
            rows.append(row)
        df = pd.DataFrame(rows)

        if test:
            print(df)
            return

        df.to_csv(log_path, index=False)

        progress.update(len(paths))

    progress.close()


def main(args):
    os.makedirs(args.output_path, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    prompt = args.prompt
    logging.info(f"Prompt: {prompt}")

    image_paths = _get_image_paths(args.img_path)
    if args.assert_images > 0:
        assert (
            len(image_paths) == args.assert_images
        ), f"Found {len(image_paths)} images, expected {args.assert_images}"

    rows = []
    if args.resume is not None:
        log_path = args.resume
        candidate_set = set()
        df_resume = pd.read_csv(log_path)
        for path in image_paths:
            if not args.full_paths:
                image_name = str(Path(path).stem)
            else:
                image_name = str(Path(path))
            if image_name not in df_resume.image.unique():
                candidate_set.add(path)

        logging.info(
            f"Resuming from {log_path}, {len(candidate_set)}/{len(image_paths)} images remaining"
        )
        image_paths = list(candidate_set)
        rows = df_resume.to_dict("records")
    else:
        prefix = None
        suffix = None
        if args.return_top_tokens:
            suffix = f"top{args.top_k}_tokens"
        prefix = prefix + "_" if prefix else ""
        suffix = "_" + suffix if suffix else ""
        output_file = f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_{prefix}{args.model}{suffix}.csv'
        log_path = os.path.join(args.output_path, output_file)

    image_paths = list(image_paths)

    if args.subsample > 0:
        random.shuffle(image_paths)
        image_paths = image_paths[: args.subsample]

    if len(image_paths) > 0:
        vl_model = create_model(args.model)

        for k, v in args.kwargs.items():
            if hasattr(vl_model, k):
                setattr(vl_model, k, v)
            else:
                raise ValueError(f"Invalid argument {k} for model {args.model}")

        eval_model(
            vl_model,
            image_paths,
            prompt,
            rows,
            log_path,
            args.test,
            args.return_top_tokens,
            args.top_k,
            args.batch_size,
            args.full_paths,
            **args.img_kwargs,
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, help="Model to use for evaluation")
    parser.add_argument(
        "--img-path",
        type=str,
        default="/workspace/data/modelvshuman/datasets/cue-conflict",
    )
    parser.add_argument(
        "--assert-images",
        type=int,
        default=1280,
        help="Assert number of images found in the image path. Default is 1280 (cue-conflict).",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="../raw-data/",
        help="Path to save the output csv file. Filename will be autogenerated.",
    )
    parser.add_argument(
        "--resume", default=None, type=str, help="Path to csv to resume from"
    )
    parser.add_argument(
        "--prompt", type=str, default="Describe the image. Keep your response short."
    )
    parser.add_argument(
        "--return-top-tokens",
        type=str2bool,
        default=False,
        help="Return the top k tokens instead of full generation. Use with --top-k",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=16,
        help="Number of tokens to return. Only used with --return-top-tokens",
    )
    parser.add_argument(
        "--batch-size",
        default=1,
        type=int,
        help="Batch size for evaluation, note that if this is not implemented in the model, it will fall back to single samle inference.",
    )
    parser.add_argument("--seed", default=0, type=int, help="Random seed.")
    parser.add_argument(
        "--test",
        action="store_true",
        default=False,
        help="Test mode. Runs a single batch without logging.",
    )
    parser.add_argument(
        "--kwargs",
        type=eval,
        default="{}",
        help="Special keyword arguments passed to the model. Useful to set temperate etc. (if supported by the model).",
    )
    parser.add_argument(
        "--img-kwargs",
        type=eval,
        default="{}",
        help="Special keyword arguments for image processing. Supports patch_size and noise_level.",
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=-1,
        help="Use a random subset of images defined by this number. Default is -1, which means use all images.",
    )
    parser.add_argument(
        "--full-paths",
        action="store_true",
        default=False,
        help="Use full paths in the log file.",
    )

    _args = parser.parse_args()

    # setup logging
    logging.basicConfig(level=logging.INFO)

    wandb.init(project="vlm_shapebias")

    main(_args)
