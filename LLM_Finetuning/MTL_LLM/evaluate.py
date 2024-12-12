# Import libraries
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# Define evaluation function
def evaluate(args, model, dataloader, device, threshold=0.5):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    all_labels_bias = []
    all_labels_stereotype = []
    all_preds_bias = []
    all_preds_stereotype = []
    
    with torch.no_grad():  # Disable gradient calculations
        for batch in dataloader:
            # Move batch data to the same device as the model
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            tasks = batch['task']
            all_labels = batch['label'].to(device)

            # Create separate lists for bias and stereotype labels
            labels_bias = []
            labels_stereotype = []
            
            for i, task in enumerate(tasks):
                if task == 0:
                    labels_bias.append(all_labels[i])
                else:
                    labels_stereotype.append(all_labels[i])

            # Convert lists to tensors
            labels_bias = torch.tensor(labels_bias, dtype=torch.float).to(device) if labels_bias else None
            labels_stereotype = torch.tensor(labels_stereotype, dtype=torch.float).to(device) if labels_stereotype else None
            # print("Batch Input ids: ", input_ids)
            # print("Batch attention mask: ", attention_mask)
            # print("Batch Labels: ", all_labels)
            # print("Batch Tasks: ", tasks)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, tasks=tasks)
            
            # Extract logits
            logits_bias = outputs['logits_bias']
            logits_stereo = outputs['logits_stereo']

            loss_bias = 0
            loss_stereo = 0
            if logits_bias is not None:
                bias_logits = logits_bias.view(-1)
                loss_bias = model.bias_loss_fn(bias_logits, labels_bias)

            if logits_stereo is not None:
                stereo_logits = logits_stereo.view(-1)
                loss_stereo = model.stereotype_loss_fn(stereo_logits, labels_stereotype)
            
            # Sum the losses
            batch_loss = args.alpha*loss_bias + (1-args.alpha)*loss_stereo
            total_loss += batch_loss.item()

            # Collect labels and predictions for bias task
            if logits_bias is not None:
                # Make predictions (1 if probability >= threshold, else 0)
                preds_bias = (logits_bias >= threshold).int().cpu()
                # print("-------------------------------------------------------")
                # print("Logits Bias: ", logits_bias)
                # print("Predictions bias: ", preds_bias)
                # print("-------------------------------------------------------")
                all_preds_bias.extend(preds_bias)
                all_labels_bias.extend(labels_bias.cpu().numpy())

            # Collect labels and predictions for stereotype task
            if logits_stereo is not None:
                preds_stereo = (logits_stereo >= threshold).int().cpu()
                # print("-------------------------------------------------------")
                # print("Logits Stereotype: ", logits_stereo)
                # print("Predictions Stereotype: ", preds_stereo)
                # print("-------------------------------------------------------")
                all_preds_stereotype.extend(preds_stereo)
                all_labels_stereotype.extend(labels_stereotype.cpu().numpy())

    # Calculate average loss
    avg_loss = total_loss / len(dataloader)

    # Compute accuracy and F1 score for bias task
    if all_labels_bias:
        accuracy_bias = accuracy_score(all_labels_bias, all_preds_bias)
        f1_bias = f1_score(all_labels_bias, all_preds_bias, average='macro')

        # Confusion Matrix and Classification Report for Bias
        cm_bias = confusion_matrix(all_labels_bias, all_preds_bias)
        report_bias = classification_report(all_labels_bias, all_preds_bias, digits=4)
        
        print("\nBias Task - Confusion Matrix:")
        print(cm_bias)
        print("\nBias Task - Classification Report:")
        print(report_bias)
    else:
        accuracy_bias, f1_bias = None, None

    # Compute accuracy and F1 score for stereotype task
    if all_labels_stereotype:
        accuracy_stereotype = accuracy_score(all_labels_stereotype, all_preds_stereotype)
        f1_stereotype = f1_score(all_labels_stereotype, all_preds_stereotype, average='macro')

        # Confusion Matrix and Classification Report for Stereotype
        cm_stereotype = confusion_matrix(all_labels_stereotype, all_preds_stereotype)
        report_stereotype = classification_report(all_labels_stereotype, all_preds_stereotype, digits=4)
        
        print("\nStereotype Task - Confusion Matrix:")
        print(cm_stereotype)
        print("\nStereotype Task - Classification Report:")
        print(report_stereotype)
    else:
        accuracy_stereotype, f1_stereotype = None, None

    return {
        'avg_loss': avg_loss,
        'accuracy_bias': accuracy_bias,
        'f1_bias': f1_bias,
        'accuracy_stereotype': accuracy_stereotype,
        'f1_stereotype': f1_stereotype
    }