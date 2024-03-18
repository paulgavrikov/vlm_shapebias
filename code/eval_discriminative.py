import torch
import os
import timm
import pandas as pd
import argparse
from tqdm import tqdm
import numpy as np
from modelvshuman.datasets.decision_mappings import DecisionMapping
import modelvshuman.helper.human_categories as hc
from shapebias_utils import CueConflictDataset


class ImageNetProbabilitiesTo16ClassesProbabilitiesMapping(DecisionMapping):
    """Return the 16 class probabilities"""

    def __init__(self, aggregation_function=None):
        if aggregation_function is None:
            aggregation_function = np.mean
        self.aggregation_function = aggregation_function
        self.categories = hc.get_human_object_recognition_categories()

    def __call__(self, probabilities):
        self.check_input(probabilities)

        aggregated_class_probabilities = []
        c = hc.HumanCategories()

        for category in self.categories:
            indices = c.get_imagenet_indices_for_category(category)
            values = np.take(probabilities, indices, axis=-1)
            aggregated_value = self.aggregation_function(values, axis=-1)
            aggregated_class_probabilities.append(aggregated_value)
        aggregated_class_probabilities = np.transpose(aggregated_class_probabilities)
        sorted_indices = np.flip(
            np.argsort(aggregated_class_probabilities, axis=-1), axis=-1
        )
        return (
            np.take(self.categories, sorted_indices, axis=-1),
            np.take(
                aggregated_class_probabilities / aggregated_class_probabilities.sum(),
                sorted_indices,
                axis=-1,
            )[0],
        )


def main(args):
    model = timm.create_model(args.model, pretrained=True)
    data_cfg = timm.data.resolve_data_config(model.pretrained_cfg)
    transform = timm.data.create_transform(**data_cfg)

    model.eval()
    model.cuda()

    dataset = CueConflictDataset(
        args.dataset,
        transform=transform,
        patch_size=args.image_shuffle,
        noise_level=args.image_noise,
    )

    decision = ImageNetProbabilitiesTo16ClassesProbabilitiesMapping()

    rows = []

    for i in tqdm(range(len(dataset))):
        img, filename, _, _ = dataset[i]
        probs = torch.nn.functional.softmax(model(img.unsqueeze(0).cuda()), dim=1)
        labels_16, probs_16 = decision(probs.detach().cpu().numpy())

        pred_class = labels_16[0][0]
        row = {
            "image": filename,
            "response": pred_class,
            "pred_label": pred_class,
            "pred_conf": probs_16[0][0],
        }
        rows.append(row)

    # write file
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"{args.model}.csv")
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)

    print(f"Results written to {output_file}")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model", type=str, default="resnet50")
    argparser.add_argument(
        "--dataset",
        type=str,
        default="/workspace/data/modelvshuman/datasets/cue-conflict",
    )
    argparser.add_argument(
        "--output-dir", type=str, default="../raw-data/inet_classifier/"
    )

    argparser.add_argument("--image-noise", type=float, default=0)
    argparser.add_argument("--image-shuffle", type=int, default=0)

    args = argparser.parse_args()

    main(args)
