import os
import nltk
import numpy as np
import pandas as pd
from uedi.evaluation import prepare_dataset
from uedi.representativeness import representativness
from uedi.rank import input_ranking, output_ranking


def check_nltk_package():
    if not os.path.isdir('./nltk_data'):
        os.makedirs('./nltk_data', exist_ok=True)

    nltk.data.path.append('./nltk_data')

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print('LookupError for corpora/stopwords, starting the download...')
        nltk.download('stopwords', download_dir='./nltk_data')

    if not os.path.exists('./nltk_data/corpora/wordnet.zip'):
        print('LookupError for corpora/wordnet, starting the download...')
        nltk.download('wordnet', download_dir='./nltk_data')

    if not os.path.exists('./nltk_data/corpora/omw-1.4.zip'):
        print('LookupError for corpora/omw-1.4, starting the download...')
        nltk.download('omw-1.4', download_dir='./nltk_data')


check_nltk_package()

# Compute input and output representativness
filename = 'data/Structured_Fodors-Zagats.csv'
columns = ['name', 'addr', 'city', 'phone', 'type', 'class']
df = pd.read_csv(filename)
df1, df2, dfi = prepare_dataset(df, columns)

input_repr, output_repr = representativness(df_s=df1, df_i=dfi)
print(f'\nSource 1 Representativness')
print(f'Input Representativness: {input_repr:0.4f}')
print(f'Output Representativness: {output_repr:0.4f}')

# Compute input ranking
input_ranks = input_ranking(df_s=df1, df_i=dfi)
idx = np.argmin(input_ranks)
print('\nThe least represented record is: ')
print(df1.iloc[idx])


# Compute output ranking
output_ranks = output_ranking(df_list=[df1, df2], df_i=dfi)
idx = np.argmin(output_ranks)
print('\nThe least represented record is: ')
print(dfi.iloc[idx])
