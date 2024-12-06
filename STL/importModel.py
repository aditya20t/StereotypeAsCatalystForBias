from transformers import AutoModelForSequenceClassification, AutoTokenizer

def get_model(model_name):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)
    return model, tokenizer