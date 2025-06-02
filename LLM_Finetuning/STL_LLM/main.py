# Import libraries
import torch
from transformers import (
    set_seed, 
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    DataCollatorWithPadding, 
    Trainer, 
    TrainingArguments,
    TrainerCallback
)
import torch.nn.functional as F
from parse import parseArguments
from dataset import get_dataset
from peft import prepare_model_for_kbit_training, get_peft_model
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from datasets import Dataset, DatasetDict
from quantize import quantize
from tqdm import tqdm

# Set random seed
torch.manual_seed(42)
set_seed(42)

# Parse arguments
args = parseArguments()

# Define device
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

# Import dataset
df_train, df_val, df_test = get_dataset(args, args.dataset)
print(df_train.head())
print('Train:', df_train.shape)
print('Validation:', df_val.shape)
print('Test:', df_test.shape)

# Compute class weights
class_weights = (1/df_train.labels.value_counts(normalize=True).sort_index()).tolist()
class_weights = torch.tensor(class_weights).to(device)
class_weights = class_weights / class_weights.sum()

# Convert to Huggingface Dataset
dataset_train = Dataset.from_pandas(df_train)
dataset_val = Dataset.from_pandas(df_val)
dataset_test = Dataset.from_pandas(df_test)


# Combine to same dataset
dataset = DatasetDict({
    'train': dataset_train,
    'validation': dataset_val,
    'test': dataset_test
})

print(dataset)

# Define quantization config
quantization_config, lora_config = quantize()

# Load model
model = AutoModelForSequenceClassification.from_pretrained(args.model, quantization_config=quantization_config, num_labels=2)


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model, add_prefix_space=True)

tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.pad_token = tokenizer.eos_token


# Tokenize the dataset
def dataset_processing(examples):
    return tokenizer(examples['Sentence'], truncation=True, max_length=512)

model.config.pad_token_id = tokenizer.eos_token_id
model.config.use_cache = False
model.config.pretraining_tp = 1
tokenized_datasets = dataset.map(dataset_processing, batched=True)
tokenized_datasets.set_format('torch')


# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Prepare model for kbit training
model = prepare_model_for_kbit_training(model)


# Get PEFT model
model = get_peft_model(model, lora_config)
print(model)

# Move the model to the device
model.to(device)

# Get performance metrics
def get_performance_metrics(df):
    y_true = df.labels
    y_pred = df.predictions
    print('Confusion Matrix:')
    print(confusion_matrix(y_true, y_pred))

    print('Classification Report:')
    print(classification_report(y_true, y_pred, digits=4))
    print("-----------------------------------------------------")



# Compute metrics
def compute_metrics(pred):
    predictions, labels = pred
    f1 = f1_score(labels, predictions, average='macro', zero_division=1)
    acc = accuracy_score(labels, predictions)
    return {
        'f1': f1,
        'accuracy': acc
    }


# Define custom trainer
class CustomTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        # Ensure label_weights is a tensor
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.args.device)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False):
        # Extract labels and convert them to long type for cross_entropy
        labels = inputs.pop("labels").long()

        # Forward pass
        outputs = model(**inputs)

        # Extract logits assuming they are directly outputted by the model
        logits = outputs.get('logits')

        # Compute custom loss with class weights for imbalanced data handling
        if self.class_weights is not None:
            loss = F.cross_entropy(logits, labels, weight=self.class_weights)
        else:
            loss = F.cross_entropy(logits, labels)

        return (loss, outputs) if return_outputs else loss
    
class PrintLossCallback(TrainerCallback):
    def on_log(self, args, state, control, **kwargs):
        if state.log_history:  # Check if log history exists and is not empty
            last_log = state.log_history[-1]
            if 'loss' in last_log:  # Check if 'loss' is in the last log entry
                print(f"Epoch {int(state.epoch)}: Loss = {last_log['loss']:.4f}")
            else:
                print(f"Epoch {int(state.epoch)}: Loss information not found in the last log.")
        else:
            print(f"Epoch {int(state.epoch)}: No log history available.")



# Define training arguments
training_args = TrainingArguments(
    output_dir = f'{args.dataset}_classification',
    learning_rate = args.lr,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size = args.batch_size,
    num_train_epochs = args.epochs,
    weight_decay = args.weight_decay,
    eval_strategy = 'epoch',
    save_strategy = 'epoch',
    logging_strategy= 'steps',
    logging_steps=20,
    logging_dir = f'{args.dataset}_classification_logs',
    load_best_model_at_end = False,
    dataloader_num_workers=4,
)


# define trainer
trainer = CustomTrainer(
    model = model,
    args = training_args,
    train_dataset = tokenized_datasets['train'],
    eval_dataset = tokenized_datasets['validation'],
    tokenizer = tokenizer,
    data_collator = data_collator,
    compute_metrics = compute_metrics,
    class_weights=class_weights,
    callbacks=[PrintLossCallback()],
)


trainer_result = trainer.train()


def make_predictions(model,df_test):


  # Convert summaries to a list
  sentences = df_test.Sentence.tolist()

  # Define the batch size
  batch_size = args.batch_size  # You can adjust this based on your system's memory capacity

  # Initialize an empty list to store the model outputs
  all_outputs = []

  # Process the sentences in batches
  for i in tqdm(range(0, len(sentences), batch_size), desc='Evaluating: '):
      # Get the batch of sentences
      batch_sentences = sentences[i:i + batch_size]

      # Tokenize the batch
      inputs = tokenizer(batch_sentences, return_tensors="pt", padding=True, truncation=True, max_length=512)

      # Move tensors to the device where the model is (e.g., GPU or CPU)
      inputs = {k: v.to('cuda' if torch.cuda.is_available() else 'cpu') for k, v in inputs.items()}

      # Perform inference and store the logits
      with torch.no_grad():
          outputs = model(**inputs)
          all_outputs.append(outputs['logits'])
  final_outputs = torch.cat(all_outputs, dim=0)
  df_test['predictions']=final_outputs.argmax(axis=1).cpu().numpy()
  df_test['predictions']=df_test['predictions']


model_save_name = args.model.split('/')[-1]

make_predictions(model,df_test)
torch.save(model.state_dict(), f'Models/STL_Models/{model_save_name}_{args.task}.pt')
print(f"Model saved as {model_save_name}.pt")

get_performance_metrics(df_test)