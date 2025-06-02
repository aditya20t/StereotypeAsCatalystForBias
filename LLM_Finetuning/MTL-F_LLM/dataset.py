from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split

# Define a function to add the column labels
def add_labels(df):
    df['labels'] = 0  # Initialize the column (optional but recommended)
    
    df.loc[(df['Bias'] == 0) & (df['Stereotype'] == 0), 'labels'] = 0
    df.loc[(df['Bias'] == 0) & (df['Stereotype'] == 1), 'labels'] = 1
    df.loc[(df['Bias'] == 1) & (df['Stereotype'] == 0), 'labels'] = 2
    df.loc[(df['Bias'] == 1) & (df['Stereotype'] == 1), 'labels'] = 3
    
    return df

# define the load dataset 
def get_dataset():
    # Load the StereoBias dataset
    df_train = pd.read_csv(f'LLM_Finetuning/Dataset/StereoBias/train.csv')
    df_val = pd.read_csv(f'LLM_Finetuning/Dataset/StereoBias/val.csv')
    df_test = pd.read_csv(f'LLM_Finetuning/Dataset/StereoBias/test.csv')

    # Pre-process the dataset
    # Define the column label (bias, stereotype) into 4 classes
    df_train = add_labels(df_train)
    df_val = add_labels(df_val)
    df_test = add_labels(df_test)

    return df_train, df_val, df_test