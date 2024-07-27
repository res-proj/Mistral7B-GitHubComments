import pandas as pd
import os
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

# the root path, change it to yours
base_path='./'

print(f"data file exist:{os.path.exists(base_path + 'Gemini-Qualitative Analysis.xlsx')}")

s_df = pd.read_excel(base_path + 'Gemini-Qualitative Analysis.xlsx', sheet_name='strong_discussions')
w_df = pd.read_excel(base_path + 'Gemini-Qualitative Analysis.xlsx', sheet_name='weak_discussions')

dfs = s_df.loc[(s_df['new'] <= 's0000000177') & (s_df['connection'] != 'self'), ['comment', 'Theme']].copy().astype('str')
dfw = w_df.loc[(w_df['new'] <= 'w0000000177') & (s_df['connection'] != 'self'), ['comment', 'Theme']].copy().astype('str')
dfc = pd.concat([dfs, dfw])

dfc.loc[dfc['Theme'] == '?', 'Theme'] = 'Unknow'

cache_dir = base_path + 'cache'
base_model = "mistralai/Mistral-7B-Instruct-v0.3"

x = pd.DataFrame()
x = x.assign(comment=dfc['comment'])
y = dfc['Theme']

over_sampler = RandomOverSampler(random_state=42)
rx, ry = over_sampler.fit_resample(x, y)

print(type(rx))
print(type(ry))

rx = rx.assign(Theme=ry)
# rx.to_csv(base_path + '/ros_test_csv.csv', mode='w+')

train_df, test_df = train_test_split(rx, test_size=0.15)
train_df, val_df = train_test_split(train_df, test_size=0.15)

dataset = DatasetDict()
dataset['train'] = Dataset.from_pandas(train_df)
dataset['validation'] = Dataset.from_pandas(val_df)
dataset['test'] = Dataset.from_pandas(test_df)
dataset.save_to_disk(base_path + 'dataset/dataset.hf')