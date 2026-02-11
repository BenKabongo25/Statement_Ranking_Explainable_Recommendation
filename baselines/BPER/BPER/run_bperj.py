from bperj import BPERJ
import numpy as np
import argparse

from baselines.BPER.utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", type=str, required=True,
                    help="Path to the dataset directory containing data.")
parser.add_argument('--save_dir', type=str, required=True,
                    help='directory to save results')

parser.add_argument('--top_k', type=int, help='select top k to evaluate, default=10', default=10)
parser.add_argument('--max_iter', type=int, help='max iteration number, default=500', default=500)
parser.add_argument('--learning_rate', type=float, help='learning rate, default=0.01', default=0.01)
parser.add_argument('--dimension', type=int, help='dimension of latent factors, default=20', default=20)
parser.add_argument('--alpha', type=float, help='reg for explanation, default=1.0', default=1.0)
parser.add_argument('--reg_rate', type=float, help='rate for regularization, default=0.01', default=0.01)
parser.add_argument('--mu_on_user', type=float, help='ratio on user for score prediction, default=0.7 (-1 means from 0, 0.1, ..., 1)', default=0.7)
args = parser.parse_args()

print('-----------------------------ARGUMENTS-----------------------------')
for arg in vars(args):
    value = getattr(args, arg)
    if value is None:
        value = ''
    print('{:30} {}'.format(arg, value))
print('-----------------------------ARGUMENTS-----------------------------')


print(now_time() + 'Program Starts')
train_tuple_list, test_tuple_list, user2items_test, user2index, item2index, exp2index = load_data(args.dataset_dir)
rec = BPERJ(train_tuple_list, len(user2index), len(item2index), len(exp2index), args.learning_rate, args.dimension, args.alpha, args.reg_rate)

for it in range(1, args.max_iter + 1):
    if it % 100 == 0:
        print(now_time() + 'iteration {}'.format(it))
    rec.train_one_epoch()

# evaluating recommendation
print(now_time() + 'evaluating recommendation performance')
user2items_top = rec.get_prediction_item(args.top_k, list(user2items_test.keys()))
ndcg, precision, recall, f1 = evaluate_item(user2items_test, user2items_top)
print(now_time() + 'NDCG on test set: {}'.format(ndcg))
print(now_time() + 'Precision on test set: {}'.format(precision))
print(now_time() + 'Recall on test set: {}'.format(recall))
print(now_time() + 'F1 on test set: {}'.format(f1))

# evaluating explanation
new_test_tuple_list = []
for x in test_tuple_list:
    u = x[0]
    i = x[1]
    items = user2items_top[u]
    if i in items:
        new_test_tuple_list.append(x)
print(now_time() + 'evaluating explanation performance on {:.2%} samples'.format(len(new_test_tuple_list) / len(test_tuple_list)))
if args.mu_on_user == -1:
    for mu in np.arange(0, 1.1, 0.1):
        print(now_time() + 'explanation results with {} on users'.format(mu))
        test_tuple_predict = rec.get_prediction_exp(args.top_k, new_test_tuple_list, mu)
        ndcg, precision, recall, f1 = evaluate_exp(new_test_tuple_list, test_tuple_predict)
        print(now_time() + 'NDCG on test set: {}'.format(ndcg))
        print(now_time() + 'Precision on test set: {}'.format(precision))
        print(now_time() + 'Recall on test set: {}'.format(recall))
        print(now_time() + 'F1 on test set: {}'.format(f1))
else:
    test_tuple_predict = rec.get_prediction_exp(args.top_k, new_test_tuple_list, args.mu_on_user)
    ndcg, precision, recall, f1 = evaluate_exp(new_test_tuple_list, test_tuple_predict)
    print(now_time() + 'NDCG on test set: {}'.format(ndcg))
    print(now_time() + 'Precision on test set: {}'.format(precision))
    print(now_time() + 'Recall on test set: {}'.format(recall))
    print(now_time() + 'F1 on test set: {}'.format(f1))