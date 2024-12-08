from transformers import AdamW
from tqdm import tqdm
from evaluateModel import evaluate_model
from saveModel import save_model

def train_model(args, model, training_dataloader, validation_dataloader, device, scaler):
    model.train()
    # Define the optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Train the model
    for epoch in range(args.epochs):
        # Loss for the epoch
        total_train_loss = 0

        print(f"========Epoch {epoch + 1}/{args.epochs}========")
        # Iterate over the training dataloader
        for batch in tqdm(training_dataloader):

            # Extract the batch
            input_ids = batch[0]
            attention_mask = batch[1]
            labels = batch[2].long()

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            loss, logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=False
            )

            # Accumulate the loss
            total_train_loss += loss.item()

            # Backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        # Calculate the average loss
        avg_train_loss = total_train_loss / len(training_dataloader)

        # Validation
        print('Running validation...')
        validation_loss, confusion_matrix, classification_report = evaluate_model(model, validation_dataloader, val=True)
        
        # Print the metrics
        print(f'Epoch: {epoch + 1}/{args.epochs}')
        print(f'Training Loss: {avg_train_loss}')
        print(f'Validation Loss: {validation_loss}')
        print(f'Macro-F1: {classification_report["macro avg"]["f1-score"]}')
        print('Confusion Matrix:')
        print(confusion_matrix)
        print('---------------------------------')

    save_model(args.model, args.dataset, model)
    return model



