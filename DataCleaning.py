combined_data['Label'].value_counts()
combined_data['attack_cat'].isnull().sum()
combined_data['attack_cat'] = combined_data['attack_cat'].fillna(value='normal').apply(lambda x: x.strip().lower())
combined_data['attack_cat'].value_counts()
combined_data['attack_cat'] = combined_data['attack_cat'].replace('backdoors','backdoor', regex=True).apply(lambda x: x.strip().lower())
combined_data['attack_cat'].value_counts()
combined_data.isnull().sum()
combined_data['ct_flw_http_mthd'] = combined_data['ct_flw_http_mthd'].fillna(value=0)
combined_data['is_ftp_login'].value_counts()
combined_data['is_ftp_login'] = combined_data['is_ftp_login'].fillna(value=0)
combined_data['is_ftp_login'].value_counts()
combined_data['is_ftp_login'] = np.where(combined_data['is_ftp_login']>1, 1, combined_data['is_ftp_login'])
combined_data['is_ftp_login'].value_counts()
combined_data['service'].value_counts()
#combined_data['service'] = combined_data['servie'].replace(to_replace='-', value='None')
combined_data['service'] = combined_data['service'].apply(lambda x:"None" if x=='-' else x)
combined_data['service'].value_counts()
combined_data['ct_ftp_cmd'].unique()
combined_data['ct_ftp_cmd'] = combined_data['ct_ftp_cmd'].replace(to_replace=' ', value=0).astype(int)
combined_data['ct_ftp_cmd'].unique()
combined_data[['service','ct_flw_http_mthd','is_ftp_login','ct_ftp_cmd','attack_cat','Label']]
combined_data['attack_cat'].nunique()
combined_data.info()
combined_data.shape
combined_data.drop(columns=['srcip','sport','dstip','dsport','Label'],inplace=True)
combined_data.info()
combined_data.shape
train, test = train_test_split(combined_data,test_size=0.2,random_state=16)
train, val = train_test_split(train,test_size=0.2,random_state=16)
train.shape
x_train, y_train = train.drop(columns=['attack_cat']), train[['attack_cat']]
x_test, y_test = test.drop(columns=['attack_cat']), test[['attack_cat']]
x_val, y_val = val.drop(columns=['attack_cat']), val[['attack_cat']]
x_train.shape, y_train.shape
x_test.shape, y_test.shape
x_val.shape, y_val.shape
train.info()
cat_col = ['proto', 'service', 'state']
num_col = list(set(x_train.columns) - set(cat_col))
# This is formatted as code