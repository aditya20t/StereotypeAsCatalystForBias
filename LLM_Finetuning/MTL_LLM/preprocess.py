# Import libraries
import pandas as pd
from datasets import Dataset

# Define the preprocess dataset function
def preprocess_dataset(df_train_1, df_val_1, df_test_1, df_train_2, df_val_2, df_test_2):
    # Assign task labels
    df_train_1['task'] = 0
    df_val_1['task'] = 0
    df_test_1['task'] = 0

    df_train_2['task'] = 1
    df_val_2['task'] = 1
    df_test_2['task'] = 1

    # Combine the datasets
    df_train = pd.concat([df_train_1, df_train_2], ignore_index=True)
    df_val = pd.concat([df_val_1, df_val_2], ignore_index=True)
    df_test = pd.concat([df_test_1, df_test_2], ignore_index=True)

    # Convert to Huggingface Dataset
    dataset_train = Dataset.from_pandas(df_train)
    dataset_val = Dataset.from_pandas(df_val)
    dataset_test = Dataset.from_pandas(df_test)

    # Combine to same dataset
    dataset = {
        'train': dataset_train,
        'validation': dataset_val,
        'test': dataset_test
    }

    return dataset
