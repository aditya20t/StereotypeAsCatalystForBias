import logging
import torch
import nltk
import numpy as np
import os
import sys
from datasets import load_dataset, load_metric
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
from tqdm import tqdm as tqdm1
# Add the parent directory of 'utils' to the Python path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'utils')))
# print(os.path.abspath(os.path.join(os.path.dirname(__file__))))

# import importlib.util
# # Define the path to the file you want to import
# module_path = 'util/arguments.py'  # Replace with the actual file path

# # Ensure the file exists
# if not os.path.isfile(module_path):
#     raise FileNotFoundError(f"File not found: {module_path}")

# # Load the module
# spec = importlib.util.spec_from_file_location(parse_args, module_path)
# module = importlib.util.module_from_spec(spec)

# # Execute the module in its own namespace
# spec.loader.exec_module(module)

import transformers
from accelerate import Accelerator
from filelock import FileLock
from transformers import set_seed
from transformers.file_utils import is_offline_mode
from util.arguments import parse_args
from multitask_model import MultitaskModel
from preprocess import convert_to_features
from multitask_data_collator import MultitaskTrainer, NLPDataCollator
from multitask_eval import multitask_eval_fn
from checkpoint_model import save_model
from pathlib import Path


logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt")


def main():
    args = parse_args()

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).

    dataset_dict = {
        # "bias_1": load_dataset(
        #     "multitask_dataloader.py",
        #     data_files={
        #         "train": "Dataset/ToxicBias/train.csv",
        #         "validation": "Dataset/ToxicBias/val.csv",
        #     },
        # ),
        "bias_2": load_dataset(
            "multitask_dataloader.py",
            data_files={
                "train": "Dataset/BABE/train.csv",
                "validation": "Dataset/BABE/val.csv",
            },
        ),
        # "bias_3": load_dataset(
        #     "multitask_dataloader.py",
        #     data_files={
        #         "train": "Dataset/BEAD/train.csv",
        #         "validation": "Dataset/BEAD/val.csv",
        #     },
        # ),
        "stereotype": load_dataset(
            "multitask_dataloader.py",
            data_files={
                "train": "Dataset/StereoSet/train.csv",
                "validation": "Dataset/StereoSet/val.csv",
            },
        ),
        # "sentiment": load_dataset(
        #     "multitask_dataloader.py",
        #     data_files={
        #         "train": "Dataset/Sentiment/train.csv",
        #         "validation": "Dataset/Sentiment/val.csv",
        #     },
        # ),
    }

    for task_name, dataset in dataset_dict.items():
        print(task_name)
        print(dataset_dict[task_name]["train"][0])
        print()

    model_names = [args.model_name_or_path] * 2
    config_files = model_names

    for idx, task_name in enumerate(["bias_2", "stereotype"]):
        model_file = Path(f"./{task_name}_model/pytorch_model.bin")
        config_file = Path(f"./{task_name}_model/config.json")
        if model_file.is_file():
            model_names[idx] = f"./{task_name}_model"

        if config_file.is_file():
            config_files[idx] = f"./{task_name}_model"
    
    print(model_names)

    multitask_model = MultitaskModel.create(
        model_name=model_names[0],
        model_type_dict={
            "bias_2": transformers.AutoModelForSequenceClassification,
            "stereotype": transformers.AutoModelForSequenceClassification,
        },
        model_config_dict={
            "bias_2": transformers.AutoConfig.from_pretrained(
                model_names[0], num_labels=2
            ),
            "stereotype": transformers.AutoConfig.from_pretrained(
                model_names[1], num_labels=2
            )
        },
        loss_weights={"bias_2": 0.5, "stereotype": 0.5}
    )


    convert_func_dict = {
        "bias_2": convert_to_features,
        "stereotype": convert_to_features
    }

    columns_dict = {
        "bias_2": ["input_ids", "attention_mask", "labels"],
        "stereotype": ["input_ids", "attention_mask", "labels"],
    }

    features_dict = {}
    for task_name, dataset in dataset_dict.items():
        features_dict[task_name] = {}
        for phase, phase_dataset in dataset.items():
            features_dict[task_name][phase] = phase_dataset.map(
                convert_func_dict[task_name],
                batched=True,
                load_from_cache_file=False,
            )
            print(
                task_name,
                phase,
                len(phase_dataset),
                len(features_dict[task_name][phase]),
            )
            features_dict[task_name][phase].set_format(
                type="torch",
                columns=columns_dict[task_name],
            )
            print(
                task_name,
                phase,
                len(phase_dataset),
                len(features_dict[task_name][phase]),
            )

    train_dataset = {
        task_name: dataset["train"] for task_name, dataset in features_dict.items()
    }

    trainer = MultitaskTrainer(
        model=multitask_model,
        args=transformers.TrainingArguments(
            output_dir=args.output_dir,
            overwrite_output_dir=True,
            learning_rate=1e-5,
            do_train=True,
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            save_steps=3280,
        ),
        data_collator=NLPDataCollator(),
        train_dataset=train_dataset,
    )
    trainer.train()
    trainer.save_model(safe_serialization=False)
    multitask_eval_fn(multitask_model, args.model_name_or_path, dataset_dict)

    save_model(args.model_name_or_path, multitask_model)



if __name__ == "__main__":
    main()
