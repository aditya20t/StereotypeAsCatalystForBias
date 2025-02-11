import transformers


def convert_to_features(example_batch, model_name="bert-large-uncased", max_length=128):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    inputs = list(example_batch["doc"])

    features = tokenizer(
        inputs,
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )
    features["labels"] = example_batch["target"]
    return features
