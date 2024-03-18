# Are Vision Language Models Texture or Shape Biased and Can We Steer Them?
Paul Gavrikov, Jovita Lukasik, Steffen Jung, Robert Geirhos, Bianca Lamm, Muhammad Jehanzeb Mirza, Margret Keuper, Janis Keuper

[![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

[ArXiv](https://arxiv.org/abs/2403.09193)


Abstract: *Vision language models (VLMs) have drastically changed the computer vision model landscape in only a few years, opening an exciting array of new applications from zero-shot image classification, over to image captioning, and visual question answering. Unlike pure vision models, they offer an intuitive way to access visual content through language prompting. The wide applicability of such models encourages us to ask whether they also align with human vision - specifically, how far they adopt human-induced visual biases through multimodal fusion, or whether they simply inherit biases from pure vision models. One important visual bias is the texture vs. shape bias, or the dominance of local over global information. In this paper, we study this bias in a wide range of popular VLMs. Interestingly, we find that VLMs are often more shape-biased than their vision encoders, indicating that visual biases are modulated to some extent through text in multimodal models. If text does indeed influence visual biases, this suggests that we may be able to steer visual biases not just through visual input but also through language: a hypothesis that we confirm through extensive experiments. For instance, we are able to steer shape bias from as low as 49% to as high as 72% through prompting alone. For now, the strong human bias towards shape (96%) remains out of reach for all tested VLMs.*


[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg

![Hero Image](assets/teaser.jpeg)


## Reproduce our results

### Setup

Please see the instructions in ENV.md for details on how to setup the environments. You will probably need multiple environments if you want to test multiple models.

Then download the cue-conflict dataset from [here](https://github.com/rgeirhos/texture-vs-shape/tree/master/stimuli/style-transfer-preprocessed-512) to your system. 

### Evaluate Models

Use `eval_vlm.py` to generate responses for your selected model. Note that this script does not perform any classification, it just prompts a model for all images and stores the output as a CSV. For example:
```bash
cd code
python eval_vlm.py --prompt "Which option best describes the image?\nA. airplane\nB. bear\nC. bicycle\nD. bird\nE. boat\nF. bottle\nG. car\nH. cat\nI. chair\nJ. clock\nK. dog\nL. elephant\nM. keyboard\nN. knife\nO. oven\nP. truck\nAnswer with the option's letter from the given choices directly." --output-path "../raw-data/vlm/vqa/" --model "llava_1_6_vicuna_7b" --img-path "./datasets/stimuli/cue-conflict/"
```
Then you can use the classification scripts `clf_vqa.py` (if you used the VQA prompt) or `clf_caption.py` (if you used the captioning prompt) to perform the classification. You can run both scripts on entire folders or individual files. This will modify the log with the classification result and generate another CSV containing a summary of all measurements including shape bias. For example:
```bash
python clf_vqa.py --file-dir ../raw-data/vqa/yyyyymmdd_hhmmss_your_vlm.csv
```

Once classified, you can also use `shapebias_utils.py` to directly evaluate log files. To annotate caption logs with an LLM as described in our paper, use `llm_judge.py`.

#### Prompts

| Type | Prompt |
|---|---|
| Captioning | Describe the image. Keep your response short. |
| VQA | Which option best describes the image?\nA. airplane\nB. bear\nC. bicycle\nD. bird\nE. boat\nF. bottle\nG. car\nH. cat\nI. chair\nJ. clock\nK. dog\nL. elephant\nM. keyboard\nN. knife\nO. oven\nP. truck\nAnswer with the option's letter from the given choices directly. |
| VQA (Shape-biased) | Identify the primary shape in the image.\nA. airplane\nB. bear\nC. bicycle\nD. bird\nE. boat\nF. bottle\nG. car\nH. cat\nI. chair\nJ. clock\nK. dog\nL. elephant\nM. keyboard\nN. knife\nO. oven\nP. truck\nAnswer with the option's letter from the given choices directly. |
| VQA (Texture-biased) | Identify the primary texture in the image.\nA. airplane\nB. bear\nC. bicycle\nD. bird\nE. boat\nF. bottle\nG. car\nH. cat\nI. chair\nJ. clock\nK. dog\nL. elephant\nM. keyboard\nN. knife\nO. oven\nP. truck\nAnswer with the option's letter from the given choices directly. |


### Automated Prompt Search

To use automatically search for prompts using Mixtral, use `llm_prompt_search.py`. Note that you have to manually set the accuracy/shape-bias for the default instruction in L81. 


## Citation 

If you find our work useful in your research, please consider citing:

```
@misc{gavrikov2024vision,
      title={Are Vision Language Models Texture or Shape Biased and Can We Steer Them?}, 
      author={Paul Gavrikov and Jovita Lukasik and Steffen Jung and Robert Geirhos and Bianca Lamm and Muhammad Jehanzeb Mirza and Margret Keuper and Janis Keuper},
      year={2024},
      eprint={2403.09193},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

### Legal
This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].