import argparse
import pandas as pd
import logging
from PIL import Image
import os
from torch.utils.data import Dataset
from utils import shuffle_image_patches, noise_image, process_image_as_tensor
from conf_utils import overconfidence, underconfidence, expected_calibration_error


IN16_CLASSES = [
    "bicycle",
    "chair",
    "truck",
    "bottle",
    "knife",
    "cat",
    "boat",
    "bird",
    "airplane",
    "clock",
    "dog",
    "oven",
    "elephant",
    "bear",
    "car",
    "keyboard",
]


class CueConflictDataset(Dataset):

    def __init__(
        self, root_dir, transform=None, patch_size: int = 0, noise_level: float = 0.0
    ):
        self.paths = []
        self.patch_size = patch_size
        self.noise_level = noise_level
        self._modify_img = patch_size > 0 or noise_level > 0
        self.transform = transform
        for p in os.listdir(root_dir):
            for img in os.listdir(os.path.join(root_dir, p)):
                if img.endswith(".png"):
                    filename = os.path.join(root_dir, p, img)
                    self.paths.append(filename)

        self.y_shapes = [
            _remove_numbers_from_string(x.split("/")[-1].split("-")[0])
            for x in self.paths
        ]
        self.y_texture = [
            _remove_numbers_from_string(x.split("/")[-1].split("-")[1])
            for x in self.paths
        ]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]

        filename = os.path.basename(path).split("/")[-1].replace(".png", "")

        img = Image.open(path).convert("RGB")
        if self._modify_img:
            if self.patch_size > 0:
                img = process_image_as_tensor(
                    img, lambda x: shuffle_image_patches(x, self.patch_size)
                )
            if self.noise_level > 0:
                img = process_image_as_tensor(
                    img, lambda x: noise_image(x, self.noise_level)
                )

        if self.transform:
            img = self.transform(img)

        return img, filename, self.y_shapes[idx], self.y_texture[idx]


def _remove_numbers_from_string(s: str) -> str:
    s = s.split("_")[-1].replace(".png", "")
    return "".join([i for i in s if not i.isdigit()])


def shapebias_from_df(df: pd.DataFrame, no_strict: bool = False) -> dict:

    df = df.copy()

    if not no_strict:
        assert len(df) == 1280, "Shape Bias response CSV must have 1280 rows"
    elif len(df) != 1280:
        logging.warning(
            f"Shape Bias response CSV should have 1280 rows, but has {len(df)} rows."
        )

    if "object_response" in df.columns:
        df = df.rename(columns={"imagename": "image", "object_response": "pred_label"})

    df["y_shape"] = df["image"].apply(
        lambda x: _remove_numbers_from_string(x.split("-")[0])
    )
    df["y_texture"] = df["image"].apply(
        lambda x: _remove_numbers_from_string(x.split("-")[1])
    )

    # remove those rows where shape = texture, i.e. no cue conflict present
    df = df.loc[df.y_shape != df.y_texture]

    # accuracy metrics
    accuracy = ((df.pred_label == df.y_texture) | (df.pred_label == df.y_shape)).mean()

    # bias metrics
    shape_ratio = (df.pred_label == df.y_shape).mean()
    texture_ratio = (df.pred_label == df.y_texture).mean()
    shape_bias = shape_ratio / (shape_ratio + texture_ratio)

    # confidence metrics
    mean_confidence = -1
    mean_underconfidence = -1
    mean_overconfidence = -1

    if "pred_conf" in df.columns:
        mean_confidence = df.pred_conf.mean()
        df["correct"] = (df.pred_label == df.y_texture) | (df.pred_label == df.y_shape)
        mean_underconfidence = underconfidence(df.correct, df.pred_conf)
        mean_overconfidence = overconfidence(df.correct, df.pred_conf)
        ece = expected_calibration_error(df.correct, df.pred_conf)

        result = {
            "accuracy": accuracy,
            "shape_bias": shape_bias,
            "shape_ratio": shape_ratio,
            "texture_ratio": texture_ratio,
            "mean_confidence": mean_confidence,
            "underconfidence": mean_underconfidence,
            "overconfidence": mean_overconfidence,
            "ece": ece,
            "num_responses": len(df),
        }
    else:
        result = {
            "accuracy": accuracy,
            "shape_bias": shape_bias,
            "shape_ratio": shape_ratio,
            "texture_ratio": texture_ratio,
            "num_responses": len(df),
        }

    return result


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("input", type=str)
    argparser.add_argument("--no-strict", action="store_true")
    args = argparser.parse_args()

    df = pd.read_csv(args.input)
    result = shapebias_from_df(df, no_strict=args.no_strict)
    print(result)
