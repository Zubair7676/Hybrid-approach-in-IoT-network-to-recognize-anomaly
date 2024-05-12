#Data encoding : find the categorical features and use OneHotEncoder

#Data Normalization : find numerical features and use StandaraScaler

#Feature Selection : SelectKBest
train
test
del train
del test
# x_train.drop(columns=['attack_cat'], inplace=True)
# test.drop(columns=['attack_cat'], inplace=True)
scaler = StandardScaler()
scaler = scaler.fit(x_train[num_col])
x_train[num_col] = scaler.transform(x_train[num_col])
x_test[num_col] = scaler.transform(x_test[num_col])
x_val[num_col] = scaler.transform(x_val[num_col])
x_train.isnull().sum()
x_train.head()
x_test.head()
type(x_train)
x_train.shape
x_test.shape
x_val.shape
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(sparse=False), cat_col)], remainder='passthrough')
x_train = np.array(ct.fit_transform(x_train))
x_test = np.array(ct.transform(x_test))
x_val = np.array(ct.transform(x_val))
y_train.info()
y_train.columns
attacks = y_train['attack_cat'].unique()
attacks
# Get unique elements and their counts
unique_values, counts = np.unique(y_train, return_counts=True)

# Print the unique values and their corresponding counts
for value, count in zip(unique_values, counts):
    print(f"Value: {value}, Count: {count}")
    # Get unique elements and their counts
unique_values, counts = np.unique(y_test, return_counts=True)

# Print the unique values and their corresponding counts
for value, count in zip(unique_values, counts):
    print(f"Value: {value}, Count: {count}")
    ct1 = ColumnTransformer(transformers=[('encoder', OneHotEncoder(categories=[attacks],sparse=False), ['attack_cat'])], remainder='passthrough')
y_train = np.array(ct1.fit_transform(y_train))
y_test = np.array(ct1.transform(y_test))
y_val = np.array(ct1.transform(y_val))
print(x_train)
print(x_test)
print(y_train)
print(y_test)
y_train.shape
y_train = y_train[:20000]
x_train.shape
x_train = x_train[:20000]
x_test.shape
x_train.reshape(-1,1).shape
x_train.shape
x_val.shape
    