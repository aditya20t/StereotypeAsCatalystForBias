# Load libraries 
import torch
from metrics import get_classification_report, get_confusion_matrix

# Define the evaluate_model function
def evaluate_model(model, dataloader, val=False):
    model.eval()
    total_val_loss = 0
    predictions, true_labels = [], []
    for batch in dataloader:
        input_ids = batch[0]
        attention_mask = batch[1]
        labels = batch[2].long()

        with torch.no_grad():
            loss, logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=False
            )

        total_val_loss += loss.item()

        # Get the predictions
        logits = logits.detach().cpu().numpy()
        labels = labels.cpu().numpy()
        predicted = logits.argmax(1)

        # Add the predictions and true labels to the list
        predictions.extend(predicted)
        true_labels.extend(labels)

    avg_val_loss = total_val_loss / len(dataloader)

    # Get the confusion matrix and classification report
    confusion_matrix = get_confusion_matrix(predictions, true_labels)
    classification_report = get_classification_report(predictions, true_labels, dict=val)

    return avg_val_loss, confusion_matrix, classification_report, predictions, true_labels