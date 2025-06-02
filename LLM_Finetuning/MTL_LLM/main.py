# Import libraries
import torch
from torch.amp import GradScaler
from dataset import get_dataset
from parse import parseArguments
from preprocess import preprocess_dataset
from quantize import quantize
from model import get_model
from transformers import set_seed
from tokenizeDataset import tokenizeDataset
from dataCollator import get_dataloader
from evaluateModel import evaluate
from trainModel import trainModel

# Set random seed
torch.manual_seed(42)
set_seed(42)

# Parse arguments
args = parseArguments()

# define device
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
scaler = GradScaler()

# Import dataset
df_train_1, df_val_1, df_test_1 = get_dataset(args.dataset_1, args.task_1)
df_train_2, df_val_2, df_test_2 = get_dataset(args.dataset_2, args.task_2)

# Print the stats of the dataset
print('Shape of dataset_1 train, val and test : ', df_train_1.shape, df_val_1.shape, df_test_1.shape)
print('Dataset_1 train value count: ', df_train_1.labels.value_counts())
print('Dataset_1 val value count: ', df_val_1.labels.value_counts())
print('Dataset_1 test value count: ', df_test_1.labels.value_counts())

print('Shape of dataset_2 train, val and test : ', df_train_2.shape, df_val_2.shape, df_test_2.shape)
print('Dataset_2 train value count: ', df_train_2.labels.value_counts())
print('Dataset_2 val value count: ', df_val_2.labels.value_counts())
print('Dataset_2 test value count: ', df_test_2.labels.value_counts())

# Preprocess the dataset for multi-task learning
dataset = preprocess_dataset(df_train_1, df_val_1, df_test_1, df_train_2, df_val_2, df_test_2)

# Get the class weights
class_weights_bias = (1/df_train_1.labels.value_counts(normalize=True).sort_index()).tolist()
class_weights_bias = torch.tensor(class_weights_bias).to(device)
class_weights_bias = class_weights_bias / class_weights_bias.sum()

class_weights_stereotype = (1/df_train_2.labels.value_counts(normalize=True).sort_index()).tolist()
class_weights_stereotype = torch.tensor(class_weights_stereotype).to(device)
class_weights_stereotype = class_weights_stereotype / class_weights_stereotype.sum()

class_weights = torch.cat([class_weights_bias, class_weights_stereotype])

# Get quantization config
quantization_config, lora_config = quantize()

# Get the multi-task model
model = get_model(args.model, quantization_config, lora_config, class_weights)
model.to(device)

# Tokenize the dataset
tokenized_train, tokenized_val, tokenized_test, tokenizer = tokenizeDataset(args, model, dataset)

# Define collate function
def collate_fn(batch):
    # Find the maximum length in the batch
    max_length = max([len(item['input_ids']) for item in batch])

    # Pad the input_ids to the maximum length
    input_ids = torch.stack([torch.tensor(item['input_ids'] + [tokenizer.pad_token_id] * (max_length - len(item['input_ids']))) for item in batch])

    attention_mask = torch.stack([torch.tensor(item['attention_mask'] + [0] * (max_length - len(item['attention_mask']))) for item in batch])

    labels = torch.tensor([item['labels'] for item in batch])

    tasks = torch.tensor([item['task'] for item in batch])
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'task': tasks,
        'label': labels
    }

# Get the dataloaders
train_loader = get_dataloader(tokenized_train, args.batch_size, collate_fn=collate_fn)
val_loader = get_dataloader(tokenized_val, args.batch_size, collate_fn=collate_fn)
test_loader = get_dataloader(tokenized_test, args.batch_size, collate_fn=collate_fn)

# Train the model
trained_model = trainModel(
    args, 
    model, 
    args.epochs, 
    args.batch_size, 
    args.lr, 
    args.weight_decay, 
    args.alpha, 
    train_loader,
    val_loader, 
    scaler, 
    device
)

# Evaluate of Test Set
print("Evaluating on test set...")
results = evaluate(args, trained_model, test_loader, device, threshold=args.threshold)
