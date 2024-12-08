from sklearn.metrics import confusion_matrix, classification_report, f1_score
import pandas as pd
import torch
import transformers

def multitask_eval_fn(multitask_model, model_name, features_dict, batch_size=8):
    preds_dict = {}
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    multitask_model.to(device)  # Ensure the model is on the correct device

    for task_name in ["bias_2", "stereotype"]:
        true_list = []
        pred_list = []
        val_len = len(features_dict[task_name]["validation"])
        acc = 0.0

        for index in range(0, val_len, batch_size):
            batch = features_dict[task_name]["validation"][
                index: min(index + batch_size, val_len)
            ]["doc"]
            labels = features_dict[task_name]["validation"][
                index: min(index + batch_size, val_len)
            ]["target"]
            inputs = tokenizer(batch, max_length=512, padding=True, truncation=True, return_tensors="pt")

            # Move inputs and labels to device
            inputs = {key: val.to(device) for key, val in inputs.items()}
            labels = torch.tensor(labels).to(device)

            # Ensure inputs and labels are in the correct shape
            if len(inputs["input_ids"].shape) == 1:
                inputs = {key: val.unsqueeze(0) for key, val in inputs.items()}
            if len(labels.shape) == 0:
                labels = labels.unsqueeze(0)

            # Forward pass to get logits
            logits = multitask_model(task_name, **inputs)

            # Ensure logits have the correct shape
            if logits.ndim == 1:
                logits = logits.unsqueeze(0)  # Ensure logits have shape [batch_size, num_classes]

            # Compute predictions
            predictions = torch.argmax(torch.softmax(logits, dim=1), dim=1).cpu().numpy()
            true_list.extend(labels.cpu().numpy())
            pred_list.extend(predictions)
            acc += sum(predictions == labels.cpu().numpy())

        acc = acc / val_len
        print(f"Task name: {task_name}")
        print(f"Accuracy: {acc}")

        # Compute confusion matrix
        confusion_matrix_result = confusion_matrix(true_list, pred_list)
        confusion_matrix_df = pd.DataFrame(confusion_matrix_result)
        print("---------------------------------Confusion Matrix------------------------------------")
        print(confusion_matrix_df)

        # Compute precision, recall, F1 score
        eval_metrics = classification_report(true_list, pred_list, output_dict=True)
        print("---------------------------------Evaluation Metrics------------------------------------")
        print(eval_metrics)

        # Print binary F1 score
        f1_score_binary = f1_score(true_list, pred_list, average='macro')
        print(f"Binary F1 score: {f1_score_binary}")

        # Save evaluation metrics to CSV
        eval_metrics_df = pd.DataFrame(eval_metrics).transpose()
        eval_metrics_df = eval_metrics_df.iloc[:, :-1]
        csv_filename = f"./results/{task_name}_{model_name}.csv"
        eval_metrics_df.to_csv(csv_filename, index=True)
        print(f"Results saved to {csv_filename}")
        print(eval_metrics_df)