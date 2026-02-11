# https://github.com/lileipisces/EXTRA/blob/master/process_sentence.py

import os
import pandas as pd
import pickle
import nltk
import re

from tqdm import tqdm


dataset_dir = "/data/common/RecommendationDatasets/StatementDatasets/Toys14/"
review_path = dataset_dir + "dataset_vS.csv"
sentence_path = dataset_dir + "tmp/EXTRA/sentences.pickle"  # path to save sentences
os.makedirs(os.path.dirname(sentence_path), exist_ok=True)

def get_sentences(string):
    string = re.sub('[:,?!\n]', '.', string)
    sentences = [sent.strip() for sent in string.split('.') if sent.strip() != '']
    return sentences


def get_sentence_attr(string):
    subj_num = 0
    noun_num = 0
    adj_num = 0
    words = string.lower().split()
    w_t_list = nltk.pos_tag(words)
    for (w, t) in w_t_list:
        if w in subj_words:
            subj_num += 1
        if t in noun_taggers:
            noun_num += 1
        if t in adj_taggers:
            adj_num += 1
    return len(words), subj_num, noun_num, adj_num


subj_words = ['i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours', 'ourselves']
noun_taggers = ['NN', 'NNP', 'NNPS', 'NNS']
adj_taggers = ['JJ', 'JJR', 'JJS']


reviews = pd.read_csv(review_path)["review"]
sentences = []
for idx, review in tqdm(enumerate(reviews), total=len(reviews), desc='Processing sentences'):
    review = str(review)
    exps = get_sentences(review)
    for exp in exps:
        word_n, subj_n, noun_n, adj_n = get_sentence_attr(exp)
        sentence = {
            'review_idx': idx,
            'exp': exp,
            'word_num': word_n,
            'subj_num': subj_n,
            'noun_num': noun_n,
            'adj_num': adj_n,
        }
        sentences.append(sentence)
    
    if idx <= 10:
        print(f"Review {idx}: {review}")
        for sent in sentences[-len(exps):]:
            print(f"  Sentence: {sent['exp']}, word_num: {sent['word_num']}, subj_num: {sent['subj_num']}, noun_num: {sent['noun_num']}, adj_num: {sent['adj_num']}")
            
pickle.dump(sentences, open(sentence_path, 'wb'))