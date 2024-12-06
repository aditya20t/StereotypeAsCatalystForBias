from datasets import load_dataset

# define the load dataset 
def get_dataset(dataset_name):
    df = load_dataset(f'cfilt/{dataset_name}')
    df_train = df['train'].to_pandas()
    df_val = df['validation'].to_pandas()
    df_test = df['test'].to_pandas()

    return df_train, df_val, df_test