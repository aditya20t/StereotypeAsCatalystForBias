# Import libraries
import torch
from dataset import get_dataset
from parse import parseArguments
from transformers import set_seed
from importModel import get_model
from torch.amp import GradScaler
from tokenizeSentence import encode_sentences
from torch.utils.data import DataLoader, TensorDataset
from trainModel import train_model
from evaluateModel import evaluate_model

# Set random seed
torch.manual_seed(42)
set_seed(42)

# Parse arguments
args = parseArguments()

# Import dataset
df_train, df_val, df_test = get_dataset()
print(df_train.head())
print('Train:', df_train.shape)
print('Validation:', df_val.shape)
print('Test:', df_test.shape)
print('Train Value Counts: ', df_train.labels.value_counts())
print('Validation Value Counts: ', df_val.labels.value_counts())
print('Test Value Counts: ', df_test.labels.value_counts())

# Define device
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
scaler = GradScaler(device=device)

# Get the model and tokenizer
model, tokenizer = get_model(args.model)
model.to(device)

# Preprocess the data and get the encodings
input_ids_train, attention_mask_train = encode_sentences(df_train['Sentence'].values, tokenizer)
input_ids_val, attention_mask_val = encode_sentences(df_val['Sentence'].values, tokenizer)
input_ids_test, attention_mask_test = encode_sentences(df_test['Sentence'].values, tokenizer)

# Convert the encodings to tensors
input_ids_train = torch.cat(input_ids_train, dim=0).to(device)
attention_mask_train = torch.cat(attention_mask_train, dim=0).to(device)
input_ids_val = torch.cat(input_ids_val, dim=0).to(device)
attention_mask_val = torch.cat(attention_mask_val, dim=0).to(device)
input_ids_test = torch.cat(input_ids_test, dim=0).to(device)
attention_mask_test = torch.cat(attention_mask_test, dim=0).to(device)

# Print encodings of sentence 0
print(df_train['Sentence'].values[0])
print(input_ids_train[0])

# Create tensor dataset
training_dataset = TensorDataset(
    input_ids_train,
    attention_mask_train,
    torch.tensor(df_train['labels'].values).to(device)
)
validation_dataset = TensorDataset(
    input_ids_val,
    attention_mask_val,
    torch.tensor(df_val['labels'].values).to(device)
)
testing_dataset = TensorDataset(
    input_ids_test,
    attention_mask_test,
    torch.tensor(df_test['labels'].values).to(device)
)

# Create a DataLoader
train_dataloader = DataLoader(
    training_dataset,
    batch_size=args.batch_size,
    shuffle=True
)

val_dataloader = DataLoader(
    validation_dataset,
    batch_size=args.batch_size,
    shuffle=False
)

test_dataloader = DataLoader(
    testing_dataset,
    batch_size=args.batch_size,
    shuffle=False
)

# Train the model
trained_model = train_model(args, model, train_dataloader, val_dataloader, device, scaler)

# Test the model
_, test_confusion_matrix, test_classification_report, bias_report, stereotype_report = evaluate_model(trained_model, test_dataloader)

# Print the test metrics
print("--------------------------------------------------")
print('Test Metrics:')
print('Confusion Matrix:')
print(test_confusion_matrix)
print('Classification Report:')
print(test_classification_report)
print('Bias Report:')
print(bias_report)  
print('Stereotype Report:')
print(stereotype_report)
print("--------------------------------------------------")

