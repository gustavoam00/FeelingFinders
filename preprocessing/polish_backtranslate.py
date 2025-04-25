import pandas as pd

pos_df = pd.read_csv("./data/backtranslated_pos.csv")
neg_df = pd.read_csv("./data/backtranslated_neg.csv")
train_df = pd.read_csv("./data/training.csv")
pos_df['id'] = pos_df['id'].astype(str)
neg_df['id'] = neg_df['id'].astype(str)
train_df['id'] = train_df['id'].astype(str)

merged_pos = pos_df.merge(train_df, on='id', suffixes=('_pos', '_train'))
pos_df_filtered = pos_df[~pos_df['id'].isin(merged_pos[merged_pos['sentence_pos'] == merged_pos['sentence_train']]['id'])]

merged_neg = neg_df.merge(train_df, on='id', suffixes=('_neg', '_train'))
neg_df_filtered = neg_df[~neg_df['id'].isin(merged_neg[merged_neg['sentence_neg'] == merged_neg['sentence_train']]['id'])]

pos_df_filtered.to_csv("./data/backtranslated_pos_filtered.csv", index=False)
neg_df_filtered.to_csv("./data/backtranslated_neg_filtered.csv", index=False)