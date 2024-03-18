import logging
from pathlib import Path
import torch
from datetime import datetime
import os
from PIL import Image
from abc import ABC, abstractmethod
import requests, base64

try:
    from transformers import logging as transformers_logging

    # disable transformers logging for partial loading (and others!)
    transformers_logging.set_verbosity_warning()
except ImportError:
    pass

# Cache dir for models
CACHE_DIR = os.environ.get("VLM_CACHE_DIR", None)

model_id_to_name = {
    "openai_vision": "GPT-4V",
    "gemini_pro_vision": "Gemini Pro Vision",
    "qwen_vl_chat": "Qwen-VL Chat",
    "qwen_vl_plus": "Qwen-VL Plus",
    "qwen_vl_max": "Qwen-VL Max",
    "llava_1_5_7b": "LLaVA v1.5 7B",
    "llava_1_5_13b": "LLaVA v1.5 13B",
    "llava_1_6_vicuna_7b": "LLaVA-NeXT 7B",
    "llava_1_6_vicuna_13b": "LLaVA-NeXT 13B",
    "llava_1_6_34b": "LLaVA-NeXT 34B",
    "moe_llava_stablelm": "MoE-LLaVA-StableLM",
    "moe_llava_qwen": "MoE-LLaVA-Qwen",
    "moe_llava_phi2": "MoE-LLaVA-Phi2",
    "llava_rlhf_1_5_7b": "LLaVA-RLHF 7B",
    "llava_rlhf_1_5_13b": "LLaVA-RLHF 13B",
    "internvl_chat": "InternVL Chat 1.1",
    "internvl_chat_1_2_plus": "InternVL Chat 1.2+",
    "emu2_chat": "Emu2-Chat",
    "instructblip_vicuna_7b": "InstructBLIP Vicuna-7B",
    "instructblip_flan_t5_xl": "InstructBLIP Flan-T5-xl",
    "cogagent_chat": "CogAgent Chat",
    "cogvlm_chat": "CogVLM Chat",
    "cogvlm_grounding_generalist": "CogVLM Grounding Generalist",
    "cogagent_vqa": "CogAgent VQA",
    "uform_gen": "UForm Gen",
    "uform_gen_chat": "UForm Gen Chat",
    "qwen_chat": "Qwen-VL Chat",
}


class VLMModel(ABC):

    @abstractmethod
    def __init__(self):
        self.temperature = None
        self.max_new_tokens = None
        self.num_beams = None
        self.top_p = None
        self.sample = None
        self.use_cache = None
        pass

    def forward_batch(self, prompts, image_paths):
        """
        Forward batch of prompts and images.
        """
        return [self.forward(p, i) for p, i in zip(prompts, image_paths)]

    @abstractmethod
    def forward(self, prompt: str, image_path: str) -> dict:
        pass

    def forward_topk_tokens(self, prompt: str, image_path: str, top_k: int) -> dict:
        raise NotImplementedError("forward_topk_tokens not implemented")


class LLaVAModel(VLMModel):

    def __init__(self, model_name, use_4bit=False):
        super().__init__()

        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path

        self.temperature = 0.2
        self.max_new_tokens = 512

        self.model_name = model_name

        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            model_path=model_name,
            model_base=None,
            model_name=get_model_name_from_path(model_name),
            device_map="cuda" if not use_4bit else None,
            load_4bit=use_4bit,
            cache_dir=CACHE_DIR,
        )

        if "rlhf" in model_name.lower():
            from transformers import LlavaLlamaForCausalLM, PeftModel

            del self.model

            dtype = torch.bfloat16
            model_path = "LLaVA-RLHF-7b-v1.5-224/sft_model"
            lora_path = "LLaVA-RLHF-7b-v1.5-224/rlhf_lora_adapter_model"
            self.model = LlavaLlamaForCausalLM.from_pretrained(
                model_path,
                device_map={"": "cuda:0"},
                torch_dtype=dtype,
            )
            self.model = PeftModel.from_pretrained(
                self.model,
                lora_path,
            )

        if "llama-2" in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "mistral" in model_name.lower():
            conv_mode = "mistral_instruct"
        elif "v1.6-34b" in model_name.lower():
            conv_mode = "chatml_direct"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        logging.info(f"Using conversation template {conv_mode}")

        self.conv_mode = conv_mode

    @staticmethod
    def get_LLaVA_1_5_7b():
        return LLaVAModel("liuhaotian/llava-v1.5-7b")

    @staticmethod
    def get_LLaVA_1_5_13b():
        return LLaVAModel("liuhaotian/llava-v1.5-13b")

    @staticmethod
    def get_LLaVA_1_6_mistral_7b():
        return LLaVAModel("liuhaotian/llava-v1.6-mistral-7b")

    @staticmethod
    def get_LLaVA_1_6_vicuna_7b():
        return LLaVAModel("liuhaotian/llava-v1.6-vicuna-7b")

    @staticmethod
    def get_LLaVA_1_6_vicuna_13b():
        return LLaVAModel("liuhaotian/llava-v1.6-vicuna-13b")

    @staticmethod
    def get_LLaVA_1_6_34b():
        return LLaVAModel("liuhaotian/llava-v1.6-34b", use_4bit=True)

    def _prepare_inputs(self, inp: str, image_path) -> tuple:
        from llava.constants import (
            IMAGE_TOKEN_INDEX,
            DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_START_TOKEN,
            DEFAULT_IM_END_TOKEN,
        )
        from llava.conversation import SeparatorStyle, conv_templates
        from llava.mm_utils import (
            process_images,
            tokenizer_image_token,
            KeywordsStoppingCriteria,
        )

        conv = conv_templates[self.conv_mode].copy()

        image = Image.open(image_path).convert("RGB")

        image_size = image.size
        image_tensor = process_images([image], self.image_processor, self.model.config)
        image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)

        if image is not None:
            # first message
            if self.model.config.mm_use_im_start_end:
                inp = (
                    DEFAULT_IM_START_TOKEN
                    + DEFAULT_IMAGE_TOKEN
                    + DEFAULT_IM_END_TOKEN
                    + "\n"
                    + inp
                )
            else:
                inp = DEFAULT_IMAGE_TOKEN + "\n" + inp
            conv.append_message(conv.roles[0], inp)
            image = None
        else:
            # later messages
            conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .to(self.model.device)
        )
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(
            keywords, self.tokenizer, input_ids
        )

        return input_ids, image_tensor, image_size, stopping_criteria

    def forward(self, prompt: str, image_path: str) -> dict:

        input_ids, image_tensor, image_size, stopping_criteria = self._prepare_inputs(
            prompt, image_path
        )

        with torch.inference_mode():
            output_ids = self.model.generate(
                inputs=input_ids,
                images=image_tensor,
                image_sizes=[image_size],
                do_sample=self.temperature > 0,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        response = self.tokenizer.batch_decode(
            output_ids, skip_special_tokens=True, skip_prompt=True
        )[0]

        return {"response": response}

    def forward_topk_tokens(self, prompt: str, image_path: str, top_k: int) -> dict:

        input_ids, image_tensor, image_size, _ = self._prepare_inputs(
            prompt, image_path
        )
        y = self.model(input_ids, images=image_tensor, image_sizes=[image_size])
        probs = torch.softmax(y.logits[0, -1], dim=0)
        topk = torch.topk(probs, k=top_k)
        top_tokens = self.tokenizer.batch_decode(topk.indices, skip_special_tokens=True)

        return {
            "top_tokens": top_tokens,
            "top_probs": topk.values.detach().cpu().numpy().tolist(),
        }


class MoE_LLaVAModel(VLMModel):

    def __init__(self, model_name):
        from moellava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from moellava.model.builder import load_pretrained_model
        from moellava.utils import disable_torch_init
        from moellava.mm_utils import get_model_name_from_path

        disable_torch_init()
        self.tokenizer, self.model, processor, _ = load_pretrained_model(
            model_name,
            None,
            get_model_name_from_path(model_name),
            device_map="cuda",
            cache_dir=CACHE_DIR,
        )
        self.model.eval()
        self.image_processor = processor["image"]

        conv_mode = None
        if "stablelm" in model_name.lower():
            conv_mode = "stablelm"
        elif "phi" in model_name.lower():
            conv_mode = "phi"
        elif "qwen" in model_name.lower():
            conv_mode = "qwen"
        else:
            raise ValueError(f"Unknown conversation {model_name}")

        self.conv_mode = conv_mode
        self.temperature = 0.2

    @staticmethod
    def get_MoE_LLaVA_StableLM():
        return MoE_LLaVAModel("LanguageBind/MoE-LLaVA-StableLM-1.6B-4e")

    @staticmethod
    def get_MoE_LLaVA_Qwen():
        return MoE_LLaVAModel("LanguageBind/MoE-LLaVA-Qwen-1.8B-4e")

    @staticmethod
    def get_MoE_LLaVA_Phi2():
        return MoE_LLaVAModel("LanguageBind/MoE-LLaVA-Phi2-2.7B-4e")

    @staticmethod
    def get_MoE_LLaVA_Phi2_384():
        return MoE_LLaVAModel("LanguageBind/MoE-LLaVA-Phi2-2.7B-4e-384")

    def forward(self, prompt: str, image_path: str) -> dict:
        from moellava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from moellava.conversation import conv_templates, SeparatorStyle
        from moellava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria

        image = Image.open(image_path).convert("RGB")

        conv = conv_templates[self.conv_mode].copy()
        image_tensor = self.image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ].to(self.model.device, dtype=torch.float16)

        inp = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = (
            tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .to(self.model.device)
        )
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(
            keywords, self.tokenizer, input_ids
        )

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=self.temperature > 0,
                temperature=self.temperature,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        outputs = self.tokenizer.decode(
            output_ids[0, input_ids.shape[1] :], skip_special_tokens=True
        ).strip()
        return {"response": outputs}


class QwenVLMModel(VLMModel):

    def __init__(self):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen-VL-Chat", trust_remote_code=True, cache_dir=CACHE_DIR
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen-VL-Chat",
            device_map="cuda",
            trust_remote_code=True,
            cache_dir=CACHE_DIR,
        ).eval()

    @staticmethod
    def get_QwenVL_Chat():
        return QwenVLMModel()

    def forward(self, prompt: str, image_path: str) -> dict:
        query = self.tokenizer.from_list_format(
            [
                {"image": image_path},
                {"text": prompt},
            ]
        )
        inputs = self.tokenizer(query, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        with torch.inference_mode():
            # add repetition_penalty to avoid https://github.com/QwenLM/Qwen-VL/issues/175
            pred = self.model.generate(**inputs, repetition_penalty=1.2)
        response = (
            self.tokenizer.decode(
                pred.cpu()[0],
                clean_up_tokenization_spaces=True,
                skip_special_tokens=True,
            )
            .split(prompt)[1]
            .strip()
        )
        return {"response": response}


class CogModel(VLMModel):

    def __init__(self, model_name):
        super().__init__()
        from transformers import AutoModelForCausalLM, LlamaTokenizer

        self.tokenizer = LlamaTokenizer.from_pretrained(
            "lmsys/vicuna-7b-v1.5", cache_dir=CACHE_DIR
        )
        self.torch_type = torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.torch_type,
            low_cpu_mem_usage=True,
            load_in_4bit=True,
            trust_remote_code=True,
            cache_dir=CACHE_DIR,
        ).eval()

    @staticmethod
    def get_CogVLMGroundingGeneralist():
        return CogModel("THUDM/cogvlm-grounding-generalist-hf")

    @staticmethod
    def get_CogVLMChat():
        return CogModel("THUDM/cogvlm-chat-hf")

    @staticmethod
    def get_CogAgentVQA():
        return CogModel("THUDM/cogagent-vqa-hf")

    @staticmethod
    def get_CogAgentChat():
        return CogModel("THUDM/cogagent-chat-hf")

    def forward(self, prompt: str, image_path: str) -> dict:
        image = Image.open(image_path).convert("RGB")

        input_by_model = self.model.build_conversation_input_ids(
            self.tokenizer, query=prompt, history=None, images=[image]
        )

        inputs = {
            "input_ids": input_by_model["input_ids"].unsqueeze(0).cuda(),
            "token_type_ids": input_by_model["token_type_ids"].unsqueeze(0).cuda(),
            "attention_mask": input_by_model["attention_mask"].unsqueeze(0).cuda(),
            "images": (
                [[input_by_model["images"][0].to(self.torch_type).cuda()]]
                if image is not None
                else None
            ),
        }
        if "cross_images" in input_by_model and input_by_model["cross_images"]:
            inputs["cross_images"] = [
                [input_by_model["cross_images"][0].to(self.torch_type)]
            ]

        # add any transformers params here.
        gen_kwargs = {
            "max_length": 2048,
            "do_sample": False,
        }  # temperature=0.9
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs["input_ids"].shape[1] :]
            response = self.tokenizer.decode(outputs[0])
            response = response.split("</s>")[0]

        return {"response": response}


class GeminiProVisionModel(VLMModel):

    def __init__(self):
        super().__init__()

        # check if the gemini api key is set
        if "GEMINI_API_KEY" not in os.environ:
            raise ValueError("GEMINI_API_KEY not set in environment")

        import google.generativeai as genai

        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        self.model = genai.GenerativeModel(model_name="gemini-pro-vision")

        self.temperature = 0.4

    @staticmethod
    def get_GeminiProVisionModel():
        return GeminiProVisionModel()

    def forward(self, prompt: str, image_path: str) -> dict:
        import mimetypes
        import google
        import google.generativeai as genai

        start = datetime.now()

        assert Path(image_path).exists(), f"Image path {image_path} does not exist"

        image_parts = [
            {
                "mime_type": mimetypes.MimeTypes().guess_type(image_path)[0],
                "data": Path(image_path).read_bytes(),
            },
        ]
        prompt_parts = [image_parts[0], "\n" + prompt]

        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

        generation_config = genai.types.GenerationConfig(
            temperature=self.temperature,
            top_p=0 if self.temperature == 0 else None,
            top_k=1 if self.temperature == 0 else None,
        )
        self.last_generation_config = generation_config

        try:
            response = self.model.generate_content(
                prompt_parts,
                safety_settings=safety_settings,
                generation_config=generation_config,
            )
            return {"response": response.text.strip()}
        except ValueError as e:
            logging.error(f"Error generating response.")
            logging.error(f"Prompt feedback: {response.prompt_feedback}")

            if response.prompt_feedback.block_reason is not None:
                return {"response": "BLOCKED"}
            else:  # cant handle this case
                raise e
        except google.api_core.exceptions.InternalServerError:
            logging.error(f"Internal server error")
            return {"response": None}


class OpenAIModel(VLMModel):

    def __init__(self):
        super().__init__()
        with open(".openai_key", "r") as f:
            self.API_KEY = f.read().strip()
        self.temperature = 0.0

    @staticmethod
    def get_OpenAIModel():
        return OpenAIModel()

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def forward(self, prompt: str, image_path: str) -> dict:
        assert Path(image_path).exists(), f"Image path {image_path} does not exist"

        base64_image = self.encode_image(image_path)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.API_KEY}",
        }

        payload = {
            "model": "gpt-4-vision-preview",
            "seed": 42,
            "temperature": self.temperature,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            "max_tokens": 300,
        }

        response = requests.post(
            "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
        )

        response_str = None
        try:
            response_str = response.json()["choices"][0]["message"]["content"]
        except:
            logging.error(f"Error generating response. {response}")

        return dict(response=response_str)


class UFormModel(VLMModel):

    def __init__(self, model_name):
        super().__init__()
        from uform.gen_model import VLMForCausalLM, VLMProcessor

        self.model = (
            VLMForCausalLM.from_pretrained(model_name, cache_dir=CACHE_DIR)
            .eval()
            .cuda()
        )
        self.processor = VLMProcessor.from_pretrained(model_name, cache_dir=CACHE_DIR)

    @staticmethod
    def get_UFormGen():
        return UFormModel("unum-cloud/uform-gen")

    @staticmethod
    def get_UFormGenChat():
        return UFormModel("unum-cloud/uform-gen-chat")

    def forward(self, prompt: str, image_path: str) -> dict:
        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(texts=[prompt], images=[image], return_tensors="pt").to(
            self.model.device
        )
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                do_sample=False,
                use_cache=True,
                max_new_tokens=128,
                eos_token_id=32001,
                pad_token_id=self.processor.tokenizer.pad_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        decoded_text = self.processor.batch_decode(output[:, prompt_len:])[0].replace(
            "<|im_end|>", ""
        )

        return {"response": decoded_text}


class InstructBlipModel(VLMModel):

    def __init__(self, model_name):
        super().__init__()
        from transformers import (
            InstructBlipProcessor,
            InstructBlipForConditionalGeneration,
        )

        self.model = (
            InstructBlipForConditionalGeneration.from_pretrained(
                model_name, cache_dir=CACHE_DIR
            )
            .eval()
            .cuda()
        )
        self.processor = InstructBlipProcessor.from_pretrained(
            model_name, cache_dir=CACHE_DIR
        )

    @staticmethod
    def get_InstructBlip_Vicuna_7B():
        return InstructBlipModel("Salesforce/instructblip-vicuna-7b")

    @staticmethod
    def get_InstructBlip_Vicuna_13B():
        return InstructBlipModel("Salesforce/instructblip-vicuna-13b")

    @staticmethod
    def get_InstructBlip_Flan_T5_XL():
        return InstructBlipModel("Salesforce/instructblip-flan-t5-xl")

    @staticmethod
    def get_InstructBlip_Flan_T5_XXL():
        return InstructBlipModel("Salesforce/instructblip-flan-t5-xxl")

    def forward(self, prompt: str, image_path: str) -> dict:
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(
            self.model.device
        )

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                do_sample=False,
                num_beams=5,
                max_length=256,
                min_length=1,
                top_p=0.9,
                repetition_penalty=1.5,
                length_penalty=1.0,
                temperature=1,
            )

        generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[
            0
        ].strip()

        return {"response": generated_text}


class Emu2ChatModel(VLMModel):
    """
    https://huggingface.co/BAAI/Emu2-Chat
    """

    def __init__(self):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            "BAAI/Emu2-Chat", cache_dir=CACHE_DIR
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            "BAAI/Emu2-Chat",
            load_in_4bit=True,
            trust_remote_code=True,
            bnb_4bit_compute_dtype=torch.float16,
            cache_dir=CACHE_DIR,
        ).eval()

    @staticmethod
    def get_Emu2Chat():
        return Emu2ChatModel()

    def forward(self, prompt: str, image_path: str) -> dict:
        query = "[<IMG_PLH>]" + prompt
        image = Image.open(image_path).convert("RGB")

        inputs = self.model.build_input_ids(
            text=[query], tokenizer=self.tokenizer, image=[image]
        )

        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image=inputs["image"].to(torch.float16),
                max_new_tokens=64,
                length_penalty=-1,
            )

        output_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        return {"response": output_text}


class InterVLMModel(VLMModel):

    def __init__(self, model_name):
        super().__init__()
        import torch
        from transformers import AutoModel
        from transformers import AutoTokenizer, CLIPImageProcessor

        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="auto",
            cache_dir=CACHE_DIR,
        ).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)
        self.image_processor = CLIPImageProcessor.from_pretrained(
            "OpenGVLab/InternVL-Chat-Chinese-V1-1", cache_dir=CACHE_DIR
        )

        self.temperature = 0

    @staticmethod
    def get_InternVLChat():
        return InterVLMModel("OpenGVLab/InternVL-Chat-Chinese-V1-1")

    @staticmethod
    def get_InternVLChat_1_2_Plus():
        return InterVLMModel("OpenGVLab/InternVL-Chat-Chinese-V1-2-Plus")

    def forward(self, prompt: str, image_path: str) -> dict:

        image = Image.open(image_path).convert("RGB")
        image = image.resize((448, 448))

        pixel_values = self.image_processor(
            images=image, return_tensors="pt"
        ).pixel_values
        pixel_values = pixel_values.to(torch.bfloat16).cuda()

        generation_config = dict(
            num_beams=1,
            max_new_tokens=512,
            do_sample=self.temperature > 0,
            temperature=self.temperature,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        with torch.inference_mode():
            response = self.model.chat(
                self.tokenizer, pixel_values, prompt, generation_config
            )

        return {"response": response}


class QwenVLAPIModel(VLMModel):

    def __init__(self, model_name):
        super().__init__()

        # check if the api key is set
        if "DASHSCOPE_API_KEY" not in os.environ:
            raise ValueError("DASHSCOPE_API_KEY not set in environment")

        self.model_name = model_name

    @staticmethod
    def get_QwenVL_Max():
        return QwenVLAPIModel("qwen-vl-max")

    @staticmethod
    def get_QwenVL_Plus():
        return QwenVLAPIModel("qwen-vl-plus")

    def forward(self, prompt: str, image_path: str) -> dict:
        from http import HTTPStatus
        import dashscope

        dashscope.api_key = os.environ.get("DASHSCOPE_API_KEY")

        messages = [
            {
                "role": "user",
                "content": [
                    {"image": Path(image_path).absolute().as_uri()},
                    {"text": prompt},
                ],
            }
        ]
        response = dashscope.MultiModalConversation.call(
            model=self.model_name, messages=messages
        )
        response_text = None
        if response.status_code == HTTPStatus.OK:
            response_text = response.output.choices[0].message.content[0]["text"]
        else:
            logging.error(f"Dashscope API Error{response.code}: {response.message}")

        return {"response": response_text}


class LLaVARLHFModel(LLaVAModel, VLMModel):

    def __init__(self, model_name, use_4bit=False, use_rlhf=True):
        VLMModel.__init__(self)
        from peft import PeftModel

        self.temperature = 0.2
        self.max_new_tokens = 512

        self.model_name = model_name

        self.tokenizer, self.model, self.image_processor, _ = (
            LLaVARLHFModel.load_pretrained_rlhf_model(
                model_path=os.path.join(CACHE_DIR, model_name, "sft_model"),
                model_base=None,
                model_name=model_name,
                device_map="cuda" if not use_4bit else None,
                load_4bit=use_4bit,
            )
        )

        if use_rlhf:
            logging.info(
                f"Loading RLHF adapter from {os.path.join(CACHE_DIR, model_name, 'rlhf_lora_adapter_model')}"
            )
            self.model = PeftModel.from_pretrained(
                self.model,
                os.path.join(CACHE_DIR, model_name, "rlhf_lora_adapter_model"),
            )

        if "llama-2" in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "mistral" in model_name.lower():
            conv_mode = "mistral_instruct"
        elif "v1.6-34b" in model_name.lower():
            conv_mode = "chatml_direct"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        logging.info(f"Using conversation template {conv_mode}")

        self.conv_mode = conv_mode

    @staticmethod
    def load_pretrained_rlhf_model(
        model_path,
        model_base,
        model_name,
        load_8bit=False,
        load_4bit=False,
        device_map="auto",
        load_bf16=False,
    ):
        import shutil
        import warnings
        from transformers import (
            AutoTokenizer,
            AutoModelForCausalLM,
            AutoConfig,
            BitsAndBytesConfig,
        )
        import torch
        from llava.model import LlavaLlamaForCausalLM
        from llava.constants import (
            DEFAULT_IMAGE_PATCH_TOKEN,
            DEFAULT_IM_START_TOKEN,
            DEFAULT_IM_END_TOKEN,
        )

        kwargs = {"device_map": device_map}

        if load_8bit:
            kwargs["load_in_8bit"] = True
        elif load_4bit:
            kwargs["load_in_4bit"] = True
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif load_bf16:
            kwargs["torch_dtype"] = torch.bfloat16
        else:
            kwargs["torch_dtype"] = torch.float16

        if "llava" in model_name.lower():
            # Load LLaVA model
            if "lora" in model_name.lower() and model_base is None:
                warnings.warn(
                    "There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged."
                )
            if "lora" in model_name.lower() and model_base is not None:
                lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                print("Loading LLaVA from base model...")
                model = LlavaLlamaForCausalLM.from_pretrained(
                    model_base,
                    low_cpu_mem_usage=True,
                    config=lora_cfg_pretrained,
                    **kwargs,
                )
                token_num, tokem_dim = (
                    model.lm_head.out_features,
                    model.lm_head.in_features,
                )
                if model.lm_head.weight.shape[0] != token_num:
                    model.lm_head.weight = torch.nn.Parameter(
                        torch.empty(
                            token_num, tokem_dim, device=model.device, dtype=model.dtype
                        )
                    )
                    model.model.embed_tokens.weight = torch.nn.Parameter(
                        torch.empty(
                            token_num, tokem_dim, device=model.device, dtype=model.dtype
                        )
                    )

                print("Loading additional LLaVA weights...")
                if os.path.exists(os.path.join(model_path, "non_lora_trainables.bin")):
                    non_lora_trainables = torch.load(
                        os.path.join(model_path, "non_lora_trainables.bin"),
                        map_location="cpu",
                    )
                else:
                    # this is probably from HF Hub
                    from huggingface_hub import hf_hub_download

                    def load_from_hf(repo_id, filename, subfolder=None):
                        cache_file = hf_hub_download(
                            repo_id=repo_id, filename=filename, subfolder=subfolder
                        )
                        return torch.load(cache_file, map_location="cpu")

                    non_lora_trainables = load_from_hf(
                        model_path, "non_lora_trainables.bin"
                    )
                non_lora_trainables = {
                    (k[11:] if k.startswith("base_model.") else k): v
                    for k, v in non_lora_trainables.items()
                }
                if any(k.startswith("model.model.") for k in non_lora_trainables):
                    non_lora_trainables = {
                        (k[6:] if k.startswith("model.") else k): v
                        for k, v in non_lora_trainables.items()
                    }
                model.load_state_dict(non_lora_trainables, strict=False)

                from peft import PeftModel

                print("Loading LoRA weights...")
                model = PeftModel.from_pretrained(model, model_path)
                print("Merging LoRA weights...")
                model = model.merge_and_unload()
                print("Model is loaded...")
            elif model_base is not None:
                # this may be mm projector only
                print("Loading LLaVA from base model...")
                if "mpt" in model_name.lower():
                    if not os.path.isfile(
                        os.path.join(model_path, "configuration_mpt.py")
                    ):
                        shutil.copyfile(
                            os.path.join(model_base, "configuration_mpt.py"),
                            os.path.join(model_path, "configuration_mpt.py"),
                        )
                    tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
                    cfg_pretrained = AutoConfig.from_pretrained(
                        model_path, trust_remote_code=True
                    )
                    model = LlavaMPTForCausalLM.from_pretrained(
                        model_base,
                        low_cpu_mem_usage=True,
                        config=cfg_pretrained,
                        **kwargs,
                    )
                else:
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_base, use_fast=False
                    )
                    cfg_pretrained = AutoConfig.from_pretrained(model_path)
                    model = LlavaLlamaForCausalLM.from_pretrained(
                        model_base,
                        low_cpu_mem_usage=True,
                        config=cfg_pretrained,
                        **kwargs,
                    )

                mm_projector_weights = torch.load(
                    os.path.join(model_path, "mm_projector.bin"), map_location="cpu"
                )
                mm_projector_weights = {
                    k: v.to(torch.float16) for k, v in mm_projector_weights.items()
                }
                model.load_state_dict(mm_projector_weights, strict=False)
            else:
                if "mpt" in model_name.lower():
                    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                    model = LlavaMPTForCausalLM.from_pretrained(
                        model_path, low_cpu_mem_usage=True, **kwargs
                    )
                else:
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_path, use_fast=False
                    )
                    model = LlavaLlamaForCausalLM.from_pretrained(
                        model_path, low_cpu_mem_usage=True, **kwargs
                    )
        else:
            # Load language model
            if model_base is not None:
                # PEFT model
                from peft import PeftModel

                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(
                    model_base,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    device_map="auto",
                )
                print(f"Loading LoRA weights from {model_path}")
                model = PeftModel.from_pretrained(model, model_path)
                print(f"Merging weights")
                model = model.merge_and_unload()
                print("Convert to FP16...")
                model.to(torch.float16)
            else:
                use_fast = False
                if "mpt" in model_name.lower():
                    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        low_cpu_mem_usage=True,
                        trust_remote_code=True,
                        **kwargs,
                    )
                else:
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_path, use_fast=False
                    )
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path, low_cpu_mem_usage=True, **kwargs
                    )

        image_processor = None

        if "llava" in model_name.lower():
            mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
            mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
            if mm_use_im_patch_token:
                tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            if mm_use_im_start_end:
                tokenizer.add_tokens(
                    [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
                )
            model.resize_token_embeddings(len(tokenizer))

            vision_tower = model.get_vision_tower()
            if not vision_tower.is_loaded:
                vision_tower.load_model()
            if load_bf16:
                vision_tower.to(device="cuda", dtype=torch.bfloat16)
            else:
                vision_tower.to(device="cuda", dtype=torch.float16)
            image_processor = vision_tower.image_processor

        if hasattr(model.config, "max_sequence_length"):
            context_len = model.config.max_sequence_length
        else:
            context_len = 2048

        return tokenizer, model, image_processor, context_len

    @staticmethod
    def get_LLaVA_RLHF_1_5_7b_SFT():
        return LLaVARLHFModel("LLaVA-RLHF-7b-v1.5-224", use_rlhf=False)

    @staticmethod
    def get_LLaVA_RLHF_1_5_7b():
        return LLaVARLHFModel("LLaVA-RLHF-7b-v1.5-224", use_rlhf=True)

    @staticmethod
    def get_LLaVA_RLHF_1_5_13b_SFT():
        return LLaVARLHFModel("LLaVA-RLHF-13b-v1.5-336", use_rlhf=False)

    @staticmethod
    def get_LLaVA_RLHF_1_5_13b():
        return LLaVARLHFModel("LLaVA-RLHF-13b-v1.5-336", use_rlhf=True)


def create_model(model_name: str) -> VLMModel:
    vl_model = None
    if model_name == "qwen_vl_chat":
        vl_model = QwenVLMModel.get_QwenVL_Chat()
    elif model_name == "qwen_vl_max":
        vl_model = QwenVLAPIModel.get_QwenVL_Max()
    elif model_name == "qwen_vl_plus":
        vl_model = QwenVLAPIModel.get_QwenVL_Plus()
    elif model_name == "llava_1_5_7b":
        vl_model = LLaVAModel.get_LLaVA_1_5_7b()
    elif model_name == "llava_1_5_13b":
        vl_model = LLaVAModel.get_LLaVA_1_5_13b()
    # elif model_name == 'llava_1_6_mistral_7b':
    #     vl_model = LLaVAModel.get_LLaVA_1_6_mistral_7b()
    elif model_name == "llava_1_6_vicuna_7b":
        vl_model = LLaVAModel.get_LLaVA_1_6_vicuna_7b()
    elif model_name == "llava_1_6_vicuna_13b":
        vl_model = LLaVAModel.get_LLaVA_1_6_vicuna_13b()
    elif model_name == "llava_1_6_34b":
        vl_model = LLaVAModel.get_LLaVA_1_6_34b()
    elif model_name == "moe_llava_stablelm":
        vl_model = MoE_LLaVAModel.get_MoE_LLaVA_StableLM()
    elif model_name == "moe_llava_qwen":
        vl_model = MoE_LLaVAModel.get_MoE_LLaVA_Qwen()
    elif model_name == "moe_llava_phi2":
        vl_model = MoE_LLaVAModel.get_MoE_LLaVA_Phi2()
    # elif model_name == 'moe_llava_phi2_384':
    #     vl_model = MoE_LLaVAModel.get_MoE_LLaVA_Phi2_384()
    elif model_name == "cogvlm_grounding_generalist":
        vl_model = CogModel.get_CogVLMGroundingGeneralist()
    elif model_name == "cogvlm_chat":
        vl_model = CogModel.get_CogVLMChat()
    elif model_name == "cogagent_vqa":
        vl_model = CogModel.get_CogAgentVQA()
    elif model_name == "cogagent_chat":
        vl_model = CogModel.get_CogAgentChat()
    elif model_name == "gemini_pro_vision":
        vl_model = GeminiProVisionModel.get_GeminiProVisionModel()
    elif model_name == "openai_vision":
        vl_model = OpenAIModel.get_OpenAIModel()
    elif model_name == "uform_gen":
        vl_model = UFormModel.get_UFormGen()
    elif model_name == "uform_gen_chat":
        vl_model = UFormModel.get_UFormGenChat()
    elif model_name == "instructblip_vicuna_7b":
        vl_model = InstructBlipModel.get_InstructBlip_Vicuna_7B()
    elif model_name == "instructblip_vicuna_13b":
        vl_model = InstructBlipModel.get_InstructBlip_Vicuna_13B()
    elif model_name == "instructblip_flan_t5_xl":
        vl_model = InstructBlipModel.get_InstructBlip_Flan_T5_XL()
    elif model_name == "instructblip_flan_t5_xxl":
        vl_model = InstructBlipModel.get_InstructBlip_Flan_T5_XXL()
    elif model_name == "emu2_chat":
        vl_model = Emu2ChatModel.get_Emu2Chat()
    elif model_name == "internvl_chat":
        vl_model = InterVLMModel.get_InternVLChat()
    elif model_name == "internvl_chat_1_2_plus":
        vl_model = InterVLMModel.get_InternVLChat_1_2_Plus()
    elif model_name == "llava_rlhf_1_5_7b":
        vl_model = LLaVARLHFModel.get_LLaVA_RLHF_1_5_7b()
    elif model_name == "llava_rlhf_1_5_7b_sft":
        vl_model = LLaVARLHFModel.get_LLaVA_RLHF_1_5_7b_SFT()
    elif model_name == "llava_rlhf_1_5_13b":
        vl_model = LLaVARLHFModel.get_LLaVA_RLHF_1_5_13b()
    elif model_name == "llava_rlhf_1_5_13b_sft":
        vl_model = LLaVARLHFModel.get_LLaVA_RLHF_1_5_13b_SFT()
    else:
        raise ValueError(f"Unknown model {model_name}")

    return vl_model
