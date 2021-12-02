# -*- coding: utf-8 -*-

""" Use torchMoji to score texts for emoji distribution.

The resulting emoji ids (0-63) correspond to the mapping
in emoji_overview.png file at the root of the torchMoji repo.

Writes the result to a csv file.
"""
from __future__ import print_function, division, unicode_literals
import example_helper
import json
import csv
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

from torchmoji.sentence_tokenizer import SentenceTokenizer
from torchmoji.model_def import torchmoji_emojis
from torchmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH

INPUT_FILE_NAME = 'sentiment140_text_preprocessed'
OUTPUT_FOLDER_PATH = '/content/drive/MyDrive/Capstone/torchMoji/examples/output'
FILE_NAME = '{}_emoji_labeled.csv'.format(INPUT_FILE_NAME)

OUTPUT_PATH = os.path.join(OUTPUT_FOLDER_PATH, FILE_NAME)

text_feather = pd.read_feather('/content/drive/MyDrive/Capstone/torchMoji/data/{}.ftr'.format(INPUT_FILE_NAME))

sentence_num = text_feather.shape[0]

def top_elements(array, k):
    ind = np.argpartition(array, -k)[-k:]
    return ind[np.argsort(array[ind])][::-1]

print('Tokenizing using dictionary from {}'.format(VOCAB_PATH))
with open(VOCAB_PATH, 'r') as f:
    vocabulary = json.load(f)


for i in tqdm(range(1000)):
    
    text_feather_subset = text_feather.iloc[i * (sentence_num // 1000): (i + 1) * (sentence_num // 1000)] 
    
    TEST_SENTENCES = list(np.array(text_feather_subset['preprocessed_text'].tolist()))
    
    print()
    print(len(TEST_SENTENCES))
    
    maxlen = text_feather_subset['preprocessed_text_len'].max()
    # maxlen = 30
    
    
    st = SentenceTokenizer(vocabulary, maxlen, ignore_sentences_with_only_custom = True)
    
    print('Loading model from {}.'.format(PRETRAINED_PATH))
    model = torchmoji_emojis(PRETRAINED_PATH)
    print(model)
    print('Running predictions.')

    tokenized, _, _ = st.tokenize_sentences(TEST_SENTENCES)
    prob = model(tokenized)
    
    for prob in [prob]:
        # Find top emojis for each sentence. Emoji ids (0-63)
        # correspond to the mapping in emoji_overview.png
        # at the root of the torchMoji repo.
        print('Writing results to {}'.format(OUTPUT_PATH))
        scores = []
        print('next_insert : {}'.format(len(tokenized)))
        
        for i, t in enumerate(tqdm(TEST_SENTENCES)):
            if i < len(tokenized):
                t_tokens = tokenized[i]
                t_score = [t]
                t_prob = prob[i]
                ind_top = top_elements(t_prob, 5)
                t_score.append(sum(t_prob[ind_top]))
                t_score.extend(ind_top)
                t_score.extend([t_prob[ind] for ind in ind_top])
                scores.append(t_score)
                print(t_score)
                
        print()
        print("CSV path : {}".format(OUTPUT_PATH))
        
        with open(OUTPUT_PATH, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=str(','), lineterminator='\n')
            writer.writerow(['Text', 'Top5%',
                            'Emoji_1', 'Emoji_2', 'Emoji_3', 'Emoji_4', 'Emoji_5',
                            'Pct_1', 'Pct_2', 'Pct_3', 'Pct_4', 'Pct_5'])
            for i, row in enumerate(scores):
                try:
                    writer.writerow(row)
                except:
                    print("Exception at row {}!".format(i))
    
output_csv = pd.read_csv(OUTPUT_PATH)
output_csv.to_feather('/content/drive/MyDrive/Capstone/torchMoji/examples/output/{}_emoji_labeled.ftr'.format(INPUT_FILE_NAME))