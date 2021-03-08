import pandas as pd
from sklearn import model_selection
import numpy as np


train = pd.read_csv("../input/tweet-sentiment-extraction/train.csv")
train = train.dropna().reset_index(drop=True)
tweet_dataset = pd.read_csv(r'../input/complete-tweet-sentiment-extraction-data/tweet_dataset.csv')


tweet_dataset = tweet_dataset.drop_duplicates('text', keep='first')
sen_dataset = tweet_dataset[['sentiment','text','new_sentiment']]
columns = ['new_sentiment','text','sentiment']
sen_dataset.columns = columns
df = pd.merge(train, sen_dataset,how='left',on=['text','sentiment']) 


df["kfold"] = -1
df = df.sample(frac=1).reset_index(drop=True)

kf = model_selection.StratifiedKFold(n_splits=5)

for fold, (trn_, val_) in enumerate(kf.split(X=df, y=df.sentiment.values)):
    print(len(trn_), len(val_))
    df.loc[val_, 'kfold'] = fold

df.to_csv("train_folds.csv", index=False)

