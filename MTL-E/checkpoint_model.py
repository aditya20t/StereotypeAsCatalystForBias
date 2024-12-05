import transformers
import torch
import os

def save_model(model_name, multitask_model):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    for task_name in ["bias", "stereotype"]:
        # Create the directory if it doesn't exist
        save_dir = f"./{task_name}_model/{model_name}"
        os.makedirs(save_dir, exist_ok=True)

        # Check if config.json does not exist and save the configuration file
        config_path = os.path.join(save_dir, "config.json")
        if not os.path.exists(config_path):
            multitask_model.taskmodels_dict[task_name].config.to_json_file(config_path)

        # Save the model state dict
        torch.save(
            multitask_model.taskmodels_dict[task_name].state_dict(),
            os.path.join(save_dir, "pytorch_model.bin")
        )

        # Save the tokenizer
        tokenizer.save_pretrained(save_dir)