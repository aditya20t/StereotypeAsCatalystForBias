# Import libraries
from transformers import AutoTokenizer

# Define tokenize dataset function
def tokenizeDataset(args, model, dataset):
    tokenizer = AutoTokenizer.from_pretrained(args.model, add_prefix_space=True)

    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    model.shared_model.config.pad_token_id = tokenizer.eos_token_id
    model.shared_model.config.use_cache = False
    model.shared_model.config.pretraining_tp = 1

    # Define a function to tokenize the inputs
    def tokenize_inputs(examples):
        # print(examples)
        tokenized_inputs = tokenizer(examples['Sentence'], truncation=True, max_length=64, padding="max_length")
        return tokenized_inputs


    tokenized_train = dataset['train'].map(tokenize_inputs, batched=True)
    tokenized_val = dataset['validation'].map(tokenize_inputs, batched=True)
    tokenized_test = dataset['test'].map(tokenize_inputs, batched=True)

    return tokenized_train, tokenized_val, tokenized_test, tokenizer