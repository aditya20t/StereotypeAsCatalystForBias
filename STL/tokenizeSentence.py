# Function that encode every sentence, add padding and return the input ids and attention mask
def encode_sentences(sentences, tokenizer):
    input_ids = []
    attention_mask = []

    # For every sentence
    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(
            sent,
            add_special_tokens=True,
            max_length=128,
            truncation=True,
            padding='max_length',
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        #Add the encoded sentence to list
        input_ids.append(encoded_dict['input_ids'])

        #Add attention mask
        attention_mask.append(encoded_dict['attention_mask'])
    return input_ids, attention_mask