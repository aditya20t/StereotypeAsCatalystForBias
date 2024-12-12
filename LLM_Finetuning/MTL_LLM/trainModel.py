# Import libraries
import torch
from transformers import AdamW
from tqdm import tqdm
from evaluate import evaluate
from plotGraphs import plot_graphs

# Define training function
def trainModel(args, model, epochs, batch_size, learning_rate, weight_decay, alpha, train_loader, validation_loader, scaler, device, threshold=0.5):
    # Set the model to training mode
    model.train()

    model_save_name = args.model.split('/')[-1]
    # Define optimizer
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), learning_rate, weight_decay=weight_decay)

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        total_loss = 0
        len_batch = None
        for batch in tqdm(train_loader):
            if len_batch is None:
                len_batch = len(batch)
            # Get input ids, attention mask, labels and tasks
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            tasks = batch['task']

            # Create separate lists for bias and stereotype labels
            labels_bias = []
            labels_stereotype = []
            
            for i, task in enumerate(tasks):
                if task == 0:
                    labels_bias.append(labels[i])
                else:
                    labels_stereotype.append(labels[i])
            
            # Convert lists to tensors
            labels_bias = torch.tensor(labels_bias, dtype=torch.float).to(device)
            labels_stereotype = torch.tensor(labels_stereotype, dtype=torch.float).to(device)
            
            # Get the logits
            logits = model(input_ids=input_ids, attention_mask=attention_mask, tasks=tasks)
            # print("Logits: ", logits)
            # print("Labels Bias: ", labels_bias)
            # print("Labels Stereotype", labels_stereotype)

            loss_bias = 0
            loss_stereo = 0
            if logits['logits_bias'] is not None:
                bias_logits = logits['logits_bias'].view(-1)
                loss_bias = model.bias_loss_fn(bias_logits, labels_bias)

            if logits['logits_stereo'] is not None:
                stereo_logits = logits['logits_stereo'].view(-1)
                loss_stereo = model.stereotype_loss_fn(stereo_logits, labels_stereotype)

            # Get weighted loss
            loss = alpha*loss_bias + (1-alpha)*loss_stereo
            total_loss += loss.item()
            # print("Bias Loss: ", loss_bias, "Stereotype Loss: ", loss_stereo)
            # Backpropagate
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # Check for gradient flow
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         print(f'{name} grad norm: {param.grad.norm()}')
            scaler.step(optimizer)
            scaler.update()

        
        print(f"Average loss for Epoch {epoch+1}: ", total_loss/len(train_loader))
        train_losses.append(total_loss/len(train_loader))
        # Evaluation phase
        eval_metrics = evaluate(args, model, validation_loader, device, threshold=threshold)
        val_losses.append(eval_metrics['avg_loss'])
        print(f"Validation loss: {eval_metrics['avg_loss']:.4f}")
        print(f"Bias Task - Accuracy: {eval_metrics['accuracy_bias']}, F1: {eval_metrics['f1_bias']}")
        print(f"Stereotype Task - Accuracy: {eval_metrics['accuracy_stereotype']}, F1: {eval_metrics['f1_stereotype']}")
        
        # Save the model if validation loss decreases or F1 improves (can be any metric)
        if eval_metrics['avg_loss'] < best_val_loss:  # You can also track other metrics
            best_val_loss = eval_metrics['avg_loss']
            best_val_f1 = (eval_metrics['f1_bias'] + eval_metrics['f1_stereotype']) / 2  # Example F1 tracking
            torch.save(model.state_dict(), f'MTL_Models/{model_save_name}_B{batch_size}_lr{learning_rate}_ep{epochs}_cp.pt')  # Save the best model
            print(f"Best model saved at epoch {epoch+1} with validation loss: {best_val_loss:.4f}")

    print(f"Training completed. Best validation loss: {best_val_loss:.4f}, Best F1: {best_val_f1:.4f}")
    torch.save(model.state_dict(), f'MTL_Models/{model_save_name}_B{batch_size}_lr{learning_rate}_ep{epochs}.pt')
    plot_graphs(train_losses, val_losses, model_save_name, batch_size, learning_rate, epochs)  # Plot the training and validation loss

    return model