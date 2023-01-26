import pandas as pd

data_train = pd.read_csv('train.zip', compression='zip', low_memory=False)
print(data_train.columns)
