import os

def save_model(task, model_name, dataset, model):
    # Define the directory path
    save_dir = f"./SavedModels/{dataset}/{task}/{model_name}/"

    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Save the model
    model.save_pretrained(save_dir)