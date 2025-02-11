# Load libraries 
import torch
from sklearn.metrics import classification_report, confusion_matrix

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

    # Compute the overall classification report and confusion matrix
    confusion_mat = confusion_matrix(true_labels, predictions)
    class_report = classification_report(true_labels, predictions, output_dict=val)

    # Convert 4-class labels back to Bias and Stereotype dimensions
    true_bias = [1 if label in [2, 3] else 0 for label in true_labels]  # Bias = 1 for labels 2 & 3
    pred_bias = [1 if label in [2, 3] else 0 for label in predictions]

    true_stereotype = [1 if label in [1, 3] else 0 for label in true_labels]  # Stereotype = 1 for labels 1 & 3
    pred_stereotype = [1 if label in [1, 3] else 0 for label in predictions]

    # print("Value of val: ", val)
    # Compute classification reports separately for Bias and Stereotype
    bias_report = classification_report(true_bias, pred_bias, digits=4, output_dict=val)
    stereotype_report = classification_report(true_stereotype, pred_stereotype, digits=4, output_dict=val)

    return avg_val_loss, confusion_mat, class_report, bias_report, stereotype_report
    # return {
    #     "avg_val_loss": avg_val_loss,
    #     "confusion_matrix": confusion_mat,
    #     "classification_report": class_report,
    #     "bias_report": bias_report,
    #     "stereotype_report": stereotype_report
    # }