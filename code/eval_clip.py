import torch
import open_clip
import pandas as pd
from transformers import AutoModel
from transformers import CLIPImageProcessor, CLIPTokenizer
from tqdm.notebook import tqdm
import argparse
import os
from shapebias_utils import IN16_CLASSES, CueConflictDataset
from models import CACHE_DIR

imagenet_classes = IN16_CLASSES

imagenet_templates = [
    "a bad photo of a {}.",
    "a photo of many {}.",
    "a sculpture of a {}.",
    "a photo of the hard to see {}.",
    "a low resolution photo of the {}.",
    "a rendering of a {}.",
    "graffiti of a {}.",
    "a bad photo of the {}.",
    "a cropped photo of the {}.",
    "a tattoo of a {}.",
    "the embroidered {}.",
    "a photo of a hard to see {}.",
    "a bright photo of a {}.",
    "a photo of a clean {}.",
    "a photo of a dirty {}.",
    "a dark photo of the {}.",
    "a drawing of a {}.",
    "a photo of my {}.",
    "the plastic {}.",
    "a photo of the cool {}.",
    "a close-up photo of a {}.",
    "a black and white photo of the {}.",
    "a painting of the {}.",
    "a painting of a {}.",
    "a pixelated photo of the {}.",
    "a sculpture of the {}.",
    "a bright photo of the {}.",
    "a cropped photo of a {}.",
    "a plastic {}.",
    "a photo of the dirty {}.",
    "a jpeg corrupted photo of a {}.",
    "a blurry photo of the {}.",
    "a photo of the {}.",
    "a good photo of the {}.",
    "a rendering of the {}.",
    "a {} in a video game.",
    "a photo of one {}.",
    "a doodle of a {}.",
    "a close-up photo of the {}.",
    "a photo of a {}.",
    "the origami {}.",
    "the {} in a video game.",
    "a sketch of a {}.",
    "a doodle of the {}.",
    "a origami {}.",
    "a low resolution photo of a {}.",
    "the toy {}.",
    "a rendition of the {}.",
    "a photo of the clean {}.",
    "a photo of a large {}.",
    "a rendition of a {}.",
    "a photo of a nice {}.",
    "a photo of a weird {}.",
    "a blurry photo of a {}.",
    "a cartoon {}.",
    "art of a {}.",
    "a sketch of the {}.",
    "a embroidered {}.",
    "a pixelated photo of a {}.",
    "itap of the {}.",
    "a jpeg corrupted photo of the {}.",
    "a good photo of a {}.",
    "a plushie {}.",
    "a photo of the nice {}.",
    "a photo of the small {}.",
    "a photo of the weird {}.",
    "the cartoon {}.",
    "art of the {}.",
    "a drawing of the {}.",
    "a photo of the large {}.",
    "a black and white photo of a {}.",
    "the plushie {}.",
    "a dark photo of a {}.",
    "itap of a {}.",
    "graffiti of the {}.",
    "a toy {}.",
    "itap of my {}.",
    "a photo of a cool {}.",
    "a photo of a small {}.",
    "a tattoo of the {}.",
]


def zeroshot_classifier(classnames, templates, tokenizer, model):
    with torch.no_grad(), torch.cuda.amp.autocast():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [
                template.format(classname) for template in templates
            ]  # format with class
            texts = tokenizer(texts).cuda()  # tokenize
            class_embeddings = model.encode_text(texts)  # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights


def main(args):
    if args.template is None:
        templates = imagenet_templates
    else:
        templates = [args.template]

    if args.provider == "open_clip":
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                args.model, pretrained=args.pretrained, cache_dir=CACHE_DIR
            )
            tokenizer = open_clip.get_tokenizer(args.model)
            model.eval().cuda()
        except Exception as e:
            print(type(e), e)
            print(open_clip.list_models())
            exit(-1)
    elif args.provider == "hf_clip":
        image_size = 448
        processor = CLIPImageProcessor(
            size={"shortest_edge": image_size},
            do_center_crop=True,
            crop_size=image_size,
        )
        model = (
            AutoModel.from_pretrained(
                args.model,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                cache_dir=CACHE_DIR,
            )
            .to("cuda")
            .eval()
        )
        tokenizer_model = CLIPTokenizer.from_pretrained(args.model, cache_dir=CACHE_DIR)
        tokenizer = lambda x: tokenizer_model(
            x, return_tensors="pt", padding=True
        ).input_ids
        preprocess = lambda x: processor(
            images=x, return_tensors="pt", padding=True
        ).pixel_values.squeeze(0)

    print(preprocess)

    dataset = CueConflictDataset(
        args.dataset,
        transform=preprocess,
        noise_level=args.image_noise,
        patch_size=args.image_shuffle,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, num_workers=2, shuffle=False
    )

    zeroshot_weights = zeroshot_classifier(
        imagenet_classes, templates=templates, tokenizer=tokenizer, model=model
    )

    rows = []

    scaling = model.logit_scale
    if scaling != 100:
        scaling = scaling.exp().item()

    print(f"Logit Scaling Factor: {scaling}")

    top1, n = 0.0, 0.0
    for i, (images, image_name, _, _) in enumerate(tqdm(loader)):
        images = images.cuda()

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = (
                scaling * image_features @ zeroshot_weights
            )  # undo temperature scaling
        probs = torch.nn.functional.softmax(logits, dim=1)

        for j in range(images.size(0)):
            row = {
                "image": image_name[j],
                "response": imagenet_classes[logits.argmax(1)[j].item()],
                "pred_label": imagenet_classes[logits.argmax(1)[j].item()],
                "pred_conf": probs[j, logits.argmax(1)[j]].item(),
            }
            rows.append(row)

        n += images.size(0)

    df = pd.DataFrame(rows)

    model_name = args.model.replace("/", "_").replace("-", "_").replace("@", "_")
    model_name += "_" + args.pretrained

    os.makedirs(args.output_dir, exist_ok=True)

    output_file = f"clip_{model_name}_{str(args.template).replace(' ', '_').replace('{}', 'CLS')}.csv"
    output_file = os.path.join(args.output_dir, output_file)
    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-shot CLIP evaluation")
    parser.add_argument(
        "--model", default="ViT-L/14@336px", type=str, help="CLIP model name"
    )
    parser.add_argument(
        "--pretrained", default="None", type=str, help="Use pretrained weights"
    )
    parser.add_argument(
        "--provider",
        default="open_clip",
        type=str,
        choices=["open_clip", "hf_clip"],
        help="Select model provider",
    )
    parser.add_argument(
        "--dataset",
        default="/workspace/data/modelvshuman/datasets/cue-conflict/",
        type=str,
        help="Path to the dataset",
    )
    parser.add_argument("--batch-size", default=16, type=int, help="Batch size")
    parser.add_argument(
        "--template", default=None, type=str, help="Template to use for zero-shot."
    )
    parser.add_argument(
        "--output-dir", default="../raw-data/clip", type=str, help="Output directory"
    )

    parser.add_argument("--image-noise", type=float, default=0)
    parser.add_argument("--image-shuffle", type=int, default=0)

    args = parser.parse_args()

    main(args)
