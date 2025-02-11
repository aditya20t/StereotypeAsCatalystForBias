import os

def save_model(model_name, model):
    # Define the directory path
    save_dir = f"./SavedModels/{model_name}/"

    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Save the model
    model.save_pretrained(save_dir)