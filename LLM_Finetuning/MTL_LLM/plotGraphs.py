#Import libraries
import matplotlib.pyplot as plt


def plot_graphs(train_losses, val_losses, model_save_name, batch_size, learning_rate, epochs):
    # Plot training and validation loss
    plt.figure(figsize=(12, 6))

    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    # Plot validation loss
    plt.subplot(1, 2, 2)
    plt.plot(val_losses, label='Validation Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'Images/{model_save_name}_B{batch_size}_lr{learning_rate}_ep{epochs}.png')  # Save the plot as an image
    plt.show()