import pandas as pd

# def read_formatted_csv(path):
#     files_df = pd.read_csv(path, delimiter='|')
#     files_df.columns = files_df.columns.str.strip()
#     files_df = files_df[['link','word']].astype(str)
#     files_df = files_df.drop(files_df.index[0]).reset_index(drop=True)
#     files_df['link'] = files_df['link'].str.strip()
#     return files_df

def read_formatted_csv(path):
    files_df = pd.read_csv(path, delimiter=',')
    files_df.columns = files_df.columns.str.strip()
    files_df = files_df[['link','word']].astype(str)
    # files_df = files_df.drop(files_df.index[0]).reset_index(drop=True)
    files_df['link'] = files_df['link'].str.strip()
    files_df['link'] = files_df['link'].map(lambda x: '/'.join(x.split('/')[-2:]))
    return files_df

def get_first_x_words(df, x):
    words_from_df_inorder = df['word'].unique().tolist()
    first_x_words = words_from_df_inorder[:x]
    return first_x_words
