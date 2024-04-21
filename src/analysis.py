import pandas as pd
import numpy as np
import argparse
from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics import precision_recall_fscore_support


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='mistral')
args = parser.parse_args()
model_name = args.model_name

def calculate_bleu4(true_strings, pred_strings):
    bleu_scores = []

    for true, pred in zip(true_strings, pred_strings):
        # Tokenize the true and predicted strings
        true_tokens = true.split()
        pred_tokens = pred.split()
        
        # Calculate BLEU-4 score for each pair of strings
        bleu_score = sentence_bleu([true_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25))
        bleu_scores.append(bleu_score)

    # Calculate the average BLEU-4 score
    avg_bleu4 = sum(bleu_scores) / len(bleu_scores)
    return avg_bleu4

def calculate_metrics_review_needed(true_values, pred_values):
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        true_values, pred_values, average='binary')
    return precision, recall, f1_score

review_needed_filename = '../results/review_needed_{}.npz'.format(model_name)
review_comments_filename = '../results/review_comment_{}.npz'.format(model_name)

review_needed_data = np.load(review_needed_filename, allow_pickle=True)['arr_0'][0]
review_comment_data = np.load(review_comments_filename, allow_pickle=True)['arr_0'][0]

true_values = review_needed_data['true values']
pred_values = review_needed_data['pred values']

true_comments = review_comment_data['true values']
pred_comments = review_comment_data['pred values']

precision, recall, f1_score = calculate_metrics_review_needed(true_values, pred_values)

print('Precision', precision)
print('Recall', recall)
print('F1 score', f1_score)