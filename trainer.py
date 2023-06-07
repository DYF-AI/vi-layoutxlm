# vi-layoutxlm：visual independent layoutxlm
# Base transformes LayoutLMv2
# @author：DYF-AI
# @date：2022/12/01
import os
from dataclasses import dataclass
from typing import Union, Optional

import numpy as np
import seq_eval
from datasets import load_dataset, Dataset, DatasetDict, load_metric
from torch.utils.data import DataLoader
from transformers import LayoutLMv2FeatureExtractor, LayoutXLMTokenizer, PreTrainedTokenizerBase, AutoTokenizer, \
    TrainingArguments, Trainer, LayoutLMv2ForTokenClassification
from transformers.utils import PaddingStrategy

from modeling_vi_layoutxlm import ViLayoutLMv2ForTokenClassification

# step1: download dataset or load local dataset
# datasets = load_dataset("nielsr/XFUN", "xfun.fr")

MODEL_PATH = "/mnt/j/model/pretrained-model/bert_torch"

train_arrow = "/mnt/j/dataset/document-intelligence/XFUND/zh/zh.train.arrow"
test_arrow = "/mnt/j/dataset/document-intelligence/XFUND/zh/zh.val.arrow"

train_dataset = Dataset.from_file(train_arrow)
test_dataset = Dataset.from_file(test_arrow)

dataset = DatasetDict({"train":train_dataset, "validation:":test_dataset})

print("dataset:", dataset)



# setp2: dataloader
feature_extractor = LayoutLMv2FeatureExtractor(apply_ocr=False)
# tokenizer = LayoutXLMTokenizer.from_pretrained("microsoft/layoutxlm-base")
tokenizer = AutoTokenizer.from_pretrained(os.path.join(MODEL_PATH, "microsoft-layoutxlm-base"))

@dataclass
class DataCollatorForTokenClassification:
    feature_extractor: LayoutLMv2FeatureExtractor
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    use_visual_backbone: bool = False

    def __call__(self, features):
        # prepare image input
        image = self.feature_extractor([feature["original_image"] for feature in features],
                                       return_tensors="pt").pixel_values
        # prepare text input
        for feature in features:
            del feature["image"]
            del feature["id"]
            del feature["original_image"]
            del feature["entities"]
            del feature["relations"]

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt"
        )
        if self.use_visual_backbone is True:
            batch["image"] = image

        return batch

data_collator = DataCollatorForTokenClassification(
    feature_extractor,
    tokenizer,
    pad_to_multiple_of=None,
    padding="max_length",
    max_length=512,
    use_visual_backbone=False
)
data_loader = DataLoader(train_dataset, batch_size=2, collate_fn=data_collator)
batch = next(iter(data_loader))
print("batch:", batch)

labels = dataset['train'].features['labels'].feature.names
print(labels)
id2label = {k:v for k,v in enumerate(labels)}
label2id = {v:k for k,v in enumerate(labels)}


# train

# Metrics
# metric = load_metric("seqeval")
metric = load_metric("seq_eval.py")
return_entity_level_metrics = True

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    results = metric.compute(predictions=true_predictions, references=true_labels)
    if return_entity_level_metrics:
        # Unpack nested dictionaries
        final_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                for n, v in value.items():
                    final_results[f"{key}_{n}"] = v
            else:
                final_results[key] = value
        return final_results
    else:
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }


args = TrainingArguments(
    output_dir="layoutxlm-finetuned-xfund-fr", # name of directory to store the checkpoints
    overwrite_output_dir=True,
    max_steps=1000, # we train for a maximum of 1,000 batches
    warmup_ratio=0.1, # we warmup a bit
    # fp16=True, # we use mixed precision (less memory consumption)
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    learning_rate=7e-5,
    remove_unused_columns=False,
    push_to_hub=False, # we'd like to push our model to the hub during training
    do_eval=True,
    eval_steps=100,
    evaluation_strategy="steps",
)


model = ViLayoutLMv2ForTokenClassification.from_pretrained(os.path.join(MODEL_PATH, "microsoft-layoutxlm-base"),
                                                         id2label=id2label,
                                                         label2id=label2id)

# Initialize our Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()