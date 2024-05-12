dfs = []
for i in range(1,5):
    path = '/content/drive/MyDrive/code/UNSW-NB15_{}.csv'  # There are 4 input csv files
    dfs.append(pd.read_csv(path.format(i), header = None))
combined_data = pd.concat(dfs).reset_index(drop=True)  # Concat all to a single df
combined_data.head()
dataset_columns = pd.read_csv('/content/drive/MyDrive/code/NUSW-NB15_features.csv',encoding='ISO-8859-1')
dataset_columns.info()
combined_data.columns = dataset_columns['Name']
combined_data.info()
combined_data.head()