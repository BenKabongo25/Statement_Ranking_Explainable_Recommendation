import pandas as pd
import pickle
import json

from tqdm import tqdm


dataset_dir = "/data/common/RecommendationDatasets/StatementDatasets/Toys14/"
review_path = dataset_dir + "dataset_vS.csv" # resulting from format_amazon.py
sentence_path = dataset_dir + "tmp/EXTRA/sentences.pickle"  # resulting from process_sentence.py
group_path = dataset_dir + "tmp/EXTRA/groups0.9.pickle"  # resulting from group_sentence.py
ID_path = dataset_dir + "tmp/EXTRA/IDs.pickle"  # path to save explanation IDs
id2exp_path = dataset_dir + "tmp/EXTRA/id2exp.json"  # path to save id2exp


reviews = pd.read_csv(review_path).to_dict(orient='index')
sentences = pickle.load(open(sentence_path, 'rb'))
exp_id_groups = pickle.load(open(group_path, 'rb'))
id2doc = {}
for group in tqdm(exp_id_groups, desc="Keeping valid explanations"):
    exp_idx = list(group)[0]  # keep one explanation in each group
    for oexp_idx in group:
        sentence = sentences[oexp_idx]
        review_idx = sentence['review_idx']
        if review_idx not in id2doc:
            review = reviews[review_idx]
            json_doc = {
                'user': review['user_id'],
                'item': review['item_id'],
                'rating': review['rating'],
                'time': review['timestamp'],
                'exp_idx': [str(exp_idx)],
                'oexp_idx': [str(oexp_idx)]
            }
            id2doc[review_idx] = json_doc
        else:
            id2doc[review_idx]['exp_idx'].append(str(exp_idx))
            id2doc[review_idx]['oexp_idx'].append(str(oexp_idx))


IDs = []
idx_set = set()
for _, doc in id2doc.items():
    IDs.append(doc)
    exp_idx = doc['exp_idx']
    oexp_idx = doc['oexp_idx']
    idx_set |= set(exp_idx) | set(oexp_idx)
pickle.dump(IDs, open(ID_path, 'wb'))


id2exp = {}
for idx, sentence in enumerate(sentences):
    idx = str(idx)
    if idx in idx_set:
        id2exp[idx] = sentence['exp']
with open(id2exp_path, 'w', encoding='utf-8') as f:
    json.dump(id2exp, f, indent=4)
    