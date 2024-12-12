from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split

# define the load dataset 
def get_dataset(dataset_name):
    # StereoSet and ToxicBias dataset
    if(dataset_name in ['StereoSet', 'ToxicBias']):
        df = load_dataset(f'cfilt/{dataset_name}')
        df_train = df['train'].to_pandas()
        df_val = df['validation'].to_pandas()
        df_test = df['test'].to_pandas()

        return df_train, df_val, df_test
    
    # Load BABE dataset
    elif(dataset_name == 'BABE'):
        df_train = pd.read_csv('../Dataset/BABE/train.csv')
        df_val = pd.read_csv('../Dataset/BABE/val.csv')
        df_test = pd.read_csv('../Dataset/BABE/test.csv')
        return df_train, df_val, df_test
    
    # Load BEAD dataset
    elif(dataset_name == 'BEAD'):
        df_train = pd.read_csv('../Dataset/BEAD/train.csv')
        df_val = pd.read_csv('../Dataset/BEAD/val.csv')
        df_test = pd.read_csv('../Dataset/BEAD/test.csv')
        return df_train, df_val, df_test