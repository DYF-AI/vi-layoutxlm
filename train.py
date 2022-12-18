# # vi-layoutxlm：visual independent layoutxlm
# # Base transformes LayoutLMv2
# # @author：DYF-AI
# # @date：2022/12/01
# import os
# from dataclasses import dataclass
# from typing import Union, Optional
#
# import numpy as np
# from datasets import Dataset, DatasetDict, load_metric
# from torch.utils.data import DataLoader
# from transformers import LayoutLMv2FeatureExtractor, AutoTokenizer, PreTrainedTokenizerBase, TrainingArguments, Trainer
# from transformers.utils import PaddingStrategy
#
# from vi_layoutxlm import ViLayoutLMv2ForTokenClassification
#
#
# @dataclass
# class DataCollatorForTokenClassification:
#     feature_extractor: LayoutLMv2FeatureExtractor
#     tokenizer: PreTrainedTokenizerBase
#     padding: Union[bool, str, PaddingStrategy] = True
#     max_length: Optional[int] = None
#     pad_to_multiple_of: Optional[int] = None
#     label_pad_token_id: int = -100
#     use_visual_backbone: bool = False
#
#     def __call__(self, features):
#         # prepare image input
#         image = self.feature_extractor([feature["original_image"] for feature in features],
#                                        return_tensors="pt").pixel_values
#         # prepare text input
#         for feature in features:
#             del feature["image"]
#             del feature["id"]
#             del feature["original_image"]
#             del feature["entities"]
#             del feature["relations"]
#
#         batch = self.tokenizer.pad(
#             features,
#             padding=self.padding,
#             max_length=self.max_length,
#             pad_to_multiple_of=self.pad_to_multiple_of,
#             return_tensors="pt"
#         )
#         if self.use_visual_backbone is True:
#             batch["image"] = image
#
#         return batch
#
# class ViLaoutXLMTrian(object):
#     def __init__(self,
#                 model_path,
#                 train_arrow,
#                 test_arrow,
#                 batch_size = 2,
#                 use_visual_backbone=False):
#         self.model_path = model_path
#         self.train_arrow = train_arrow
#         self.test_arrow = test_arrow
#         self.batch_size = batch_size
#         self.use_visual_backbone = use_visual_backbone
#
#         self.load_dataset()
#         self.cal_id2label()
#         self.load_metric()
#         self.feature_extractor = self.get_feature_extractor()
#         self.tokenizer = self.get_tokenizer()
#         # self.get_dataloader()
#
#         self.data_collator = DataCollatorForTokenClassification(
#             self.feature_extractor,
#             self.tokenizer,
#             pad_to_multiple_of=None,
#             padding="max_length",
#             max_length=512,
#             use_visual_backbone=False
#         )
#
#     def load_dataset(self):
#         self.train_dataset = Dataset.from_file(self.train_arrow)
#         self.test_dataset = Dataset.from_file(self.test_arrow)
#         self.dataset = DatasetDict({"train": self.train_dataset, "validation:": self.test_dataset})
#
#     def get_feature_extractor(self):
#         feature_extractor = LayoutLMv2FeatureExtractor(apply_ocr=False)
#         return feature_extractor
#
#     def get_tokenizer(self):
#         tokenizer = AutoTokenizer.from_pretrained(os.path.join(MODEL_PATH, "microsoft-layoutxlm-base"))
#         return tokenizer
#
#     # def get_dataloader(self):
#     #     self.data_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=self.data_collator)
#
#
#     def cal_id2label(self):
#         labels = self.dataset['train'].features['labels'].feature.names
#         self.id2label = {k: v for k, v in enumerate(labels)}
#         self.label2id = {v: k for k, v in enumerate(labels)}
#
#
#     def load_metric(self):
#         self.metric = load_metric("seqeval_loacal.py")
#         self.return_entity_level_metrics = True
#
#     def compute_metrics(self, p):
#         predictions, labels = p
#         predictions = np.argmax(predictions, axis=2)
#         # Remove ignored index (special tokens)
#         true_predictions = [
#             [self.id2label[p] for (p, l) in zip(prediction, label) if l != -100]
#             for prediction, label in zip(predictions, labels)
#         ]
#         true_labels = [
#             [self.id2label[l] for (p, l) in zip(prediction, label) if l != -100]
#             for prediction, label in zip(predictions, labels)
#         ]
#         results = self.metric.compute(predictions=true_predictions, references=true_labels)
#         if self.return_entity_level_metrics:
#             # Unpack nested dictionaries
#             final_results = {}
#             for key, value in results.items():
#                 if isinstance(value, dict):
#                     for n, v in value.items():
#                         final_results[f"{key}_{n}"] = v
#                 else:
#                     final_results[key] = value
#             return final_results
#         else:
#             return {
#                 "precision": results["overall_precision"],
#                 "recall": results["overall_recall"],
#                 "f1": results["overall_f1"],
#                 "accuracy": results["overall_accuracy"],
#             }
#
#     def get_trian_arguments(self):
#         args = TrainingArguments(
#             output_dir="layoutxlm-finetuned-xfund-fr",  # name of directory to store the checkpoints
#             overwrite_output_dir=True,
#             max_steps=1000,  # we train for a maximum of 1,000 batches
#             warmup_ratio=0.1,  # we warmup a bit
#             # fp16=True, # we use mixed precision (less memory consumption)
#             per_device_train_batch_size=self.batch_size,
#             per_device_eval_batch_size=self.batch_size,
#             learning_rate=7e-5,
#             remove_unused_columns=False,
#             push_to_hub=False,  # we'd like to push our model to the hub during training
#             do_eval=True,
#             eval_steps=100,
#             evaluation_strategy="steps",
#         )
#         return args
#
#     def build_model(self):
#         model = ViLayoutLMv2ForTokenClassification.from_pretrained(os.path.join(MODEL_PATH, "microsoft-layoutxlm-base"),
#                                                                    id2label=self.id2label,
#                                                                    label2id=self.label2id)
#         return model
#
#     def get_trianer(self):
#         self.data_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=self.data_collator)
#         # Initialize our Trainer
#         model = self.build_model()
#         args = self.get_trian_arguments()
#         tokenizer = self.get_tokenizer()
#         trainer = Trainer(
#             model=model,
#             args=args,
#             train_dataset=self.train_dataset,
#             eval_dataset=self.test_dataset,
#             tokenizer=tokenizer,
#             data_collator=self.data_loader,
#             compute_metrics=self.compute_metrics,
#         )
#         return trainer
#
#     def train(self):
#         trainer = self.get_trianer()
#         trainer.train()
#
#
# if __name__ == "__main__":
#     MODEL_PATH = "/mnt/j/model/pretrained-model/bert_torch"
#     train_arrow = "/mnt/j/dataset/document-intelligence/XFUND/zh/zh.train.arrow"
#     test_arrow = "/mnt/j/dataset/document-intelligence/XFUND/zh/zh.val.arrow"
#
#     vilayoutxlm = ViLaoutXLMTrian(MODEL_PATH, train_arrow, test_arrow, batch_size=2, use_visual_backbone=False)
#
#     vilayoutxlm.train()