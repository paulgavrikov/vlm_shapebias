from datasets import load_dataset, Image

dataset = load_dataset("/Users/paul/Downloads/freq-cue-conflict")
split = dataset["test"]

def _remove_numbers_from_string(s: str) -> str:
    s = s.split("_")[-1].replace(".JPEG", "")
    return "".join([i for i in s if not i.isdigit()])

lf_labels = []
hf_labels = []

for row in split.cast_column("image", Image(decode=False))["image"]:
    file_name = row["path"].split("/")[-1]
    lf_labels.append(_remove_numbers_from_string(file_name).split("-")[0])
    hf_labels.append(_remove_numbers_from_string(file_name).split("-")[1])

split = split.remove_columns("label")
split = split.add_column("lf_label", lf_labels)
split = split.add_column("hf_label", hf_labels)

dataset["test"] = split

dataset.push_to_hub("paulgavrikov/frequency-cue-conflict")
