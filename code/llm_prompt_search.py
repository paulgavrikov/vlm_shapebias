import pandas as pd
import os
from argparse import ArgumentParser
from datetime import datetime
import torch
from models import CACHE_DIR
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import wandb
import logging
from models import create_model
from eval_vlm import _get_image_paths, eval_model
from utils import str2bool
from clf_vqa import process_csv

try:
    import coloredlogs

    coloredlogs.install()
except ImportError:
    pass


DEFAULT_INSTRUCTION = \
"""Your task is to help me design an ideal image classification prompt for a vision/language model to MAXIMIZE the shape bias without significantly hurting accuracy. The provided photos show natural objects modified to contain shape and texture of conflicting object classes. E.g., it could be a photo of an elephant (shape) with dog fur (texture). The model should classify the image as 'elephant' based on the shape of the object and ignore the texture. The model's accuracy is the percentage of correctly classified images. The shape bias is the ratio of how often the model classified based on shape over texture.
You can test your prompt by outputting a single new line starting with 'PROMPT: '. Do not list options - the system will provide them automatically. Try to keep the prompt as short and simple as possible but be creative. It might be reasonable to summarize insights of previous attempts and to outline your goals before responding with a new prompt, but make sure that only the prompt starts with 'PROMPT:'. In response to the prompt you will be told the accuracy and shape bias. Then you will refine the prompt and we will continue until I say stop.
Let's go!"""


def generate_next_prompt(
    model, tokenizer, conversation, temperature: float = 1.0, do_sample: bool = False
):
    clf_prompt = (
        lambda x: x
        + " Options:\nA. airplane\nB. bear\nC. bicycle\nD. bird\nE. boat\nF. bottle\nG. car\nH. cat\nI. chair\nJ. clock\nK. dog\nL. elephant\nM. keyboard\nN. knife\nO. oven\nP. truck\nAnswer with the option's letter from the given choices directly."
    )

    inputs = tokenizer.apply_chat_template(
        conversation, return_tensors="pt", add_generation_prompt=True
    )
    with torch.inference_mode():
        generated_ids = model.generate(
            inputs,
            max_new_tokens=750,
            temperature=temperature,
            do_sample=do_sample,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(
        generated_ids[0][len(inputs[0]) :],
        skip_special_tokens=True,
        clean_up_tokenization_space=True,
    )
    conversation.append({"role": "assistant", "content": response})

    logging.info(f"Response: {response}")

    vlm_prompt = None
    prompt_prefix = None
    for line in response.split("\n"):
        if line.startswith("PROMPT:"):
            prompt_prefix = line.replace("PROMPT:", "").strip()
            vlm_prompt = clf_prompt(prompt_prefix)
            logging.info(f"Prompt: {vlm_prompt}")
            break

    return prompt_prefix, vlm_prompt


def initial_conversation(args):
    conversation = [
        {"role": "user", "content": args.instruction},
        {
            "role": "assistant",
            "content": "PROMPT: Which option best describes the image?",
        },
        {
            "role": "user",
            "content": "Prompt: 'Which option best describes the image?', Accuracy: 82.58 %, Shape Bias: 59.43 %. What is your next prompt?",
        },  # TODO: this is hardcoded
    ]

    if args.load_conv is not None:
        conversation = pd.read_csv(args.load_conv).to_dict(orient="records")
        # remove non-executed assistant prompt
        if conversation[-1]["role"] == "assistant":
            conversation.pop()
        if args.instruction is not None:
            logging.warning(
                f"Loaded conversation from {args.load_conv}, but instruction was set - ignoring it!"
            )

    return conversation


def main(args):
    image_paths = list(_get_image_paths(args.img_path))
    if args.subset is not None:
        # shuffle paths
        random.shuffle(image_paths)
        image_paths = image_paths[: args.subset]

    torch.set_default_device("cuda")

    llm_tokenizer = AutoTokenizer.from_pretrained(
        args.llm, trust_remote_code=True, cache_dir=CACHE_DIR
    )
    llm_model = AutoModelForCausalLM.from_pretrained(
        args.llm,
        torch_dtype=torch.float16,
        bnb_4bit_compute_dtype=torch.float16,
        device_map="auto",
        attn_implementation="flash_attention_2",
        load_in_4bit=True,
        load_in_8bit=False,
        trust_remote_code=True,
        cache_dir=CACHE_DIR,
    )

    vlm_model = create_model(args.model)
    vlm_model.temperature = 0.0

    conversation = initial_conversation(args)

    is_running = True

    while is_running:
        try:
            prompt_prefix, vlm_prompt = generate_next_prompt(
                llm_model,
                llm_tokenizer,
                conversation,
                temperature=args.llm_temperature,
                do_sample=args.llm_sampling,
            )
            pd.DataFrame(conversation).to_csv(
                os.path.join(output_path, "conversation.csv"), index=False
            )
            if vlm_prompt is None:
                conversation.append(
                    {
                        "role": "user",
                        "content": f"No prompt found. Please provide a prompt starting with 'PROMPT:'",
                    }
                )
                pd.DataFrame(conversation).to_csv(
                    os.path.join(output_path, "conversation.csv"), index=False
                )
                continue

            output_file = f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_{args.model}.csv'
            log_path = os.path.join(output_path, output_file)
            eval_model(
                vlm_model,
                image_paths,
                vlm_prompt,
                [],
                log_path,
                test=False,
                return_top_tokens=False,
                top_k=0,
                batch_size=args.batch_size,
            )
            shape_bias_result = process_csv(
                csv_path=log_path, no_strict=args.subset is not None
            )

            shape_bias_result["prompt"] = vlm_prompt
            wandb.log(shape_bias_result)

            logging.info(f"Result: {shape_bias_result}")

            conversation.append(
                {
                    "role": "user",
                    "content": f"Prompt: '{prompt_prefix}', Accuracy: {shape_bias_result['accuracy'] * 100:.2f} %, Shape Bias: {shape_bias_result['shape_bias'] * 100:.1f} %. Can you improve this?",
                }
            )
            pd.DataFrame(conversation).to_csv(
                os.path.join(output_path, "conversation.csv"), index=False
            )
        except KeyboardInterrupt:
            is_running = False
            logging.info("Interrupted by user.")
            break


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument(
        "--img-path",
        type=str,
        default="/workspace/data/modelvshuman/datasets/cue-conflict",
    )
    parser.add_argument("--output-path", type=str, default="../raw-data/prompt_search")
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--instruction", type=str, default=DEFAULT_INSTRUCTION)
    parser.add_argument(
        "--llm", default="mistralai/Mixtral-8x7B-Instruct-v0.1", type=str
    )
    parser.add_argument(
        "--llm_sampling",
        type=str2bool,
        default=True,
        help="Use sampling instead of greedy decoding for LLM",
    )
    parser.add_argument(
        "--llm_temperature",
        default=0.2,
        type=float,
        help="Temperature for LLM sampling",
    )
    parser.add_argument(
        "--load_conv",
        type=str,
        default=None,
        help="Load a conversation from a CSV file",
    )
    parser.add_argument(
        "--subset", type=int, default=None, help="Only use a subset of the images"
    )
    argparser.add_argument(
        "--log-level",
        type=int,
        default=logging.INFO,
        help="Verbosity level for logging",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Set seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.os.path.join(args.output_path, run_id)
    wandb.init(project="llm_shape_prompt_search", config=args, name=run_id)

    os.makedirs(output_path, exist_ok=True)

    logging_file = os.path.join(output_path, "llm_sweep.log")
    logging.basicConfig(level=args.log_level, filename=logging_file)
    logging.getLogger().addHandler(logging.StreamHandler())

    logging.info(f"Arguments: {args}")

    main(args)
