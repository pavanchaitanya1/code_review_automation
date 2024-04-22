import pandas as pd
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics import precision_recall_fscore_support
from nltk.tokenize import word_tokenize

def calculate_bleu4(true_strings, pred_strings):
    bleu_scores = []
    for true, pred in zip(true_strings, pred_strings):
        true_tokens = word_tokenize(true.lower())
        pred_tokens = word_tokenize(pred.lower())
        bleu_score = sentence_bleu([true_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25))
        bleu_scores.append(bleu_score)
    return np.mean(bleu_scores)

def calculate_metrics_review_needed(true_values, pred_values):
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        true_values, pred_values, average='binary')
    return precision, recall, f1_score

models = ['GPT', 'perplexity', 'mistral']
run_types = ['', '_no_rag', '_no_yes_no']

data1 = []

for model in models:
    for run_type in run_types:
        data = []
        data.append(model)
        data.append(run_type)
        
        review_needed_filename = '../results/review_needed_{}{}.npz'.format(model, run_type)
        review_comments_filename = '../results/review_comment_{}{}.npz'.format(model, run_type)

        review_needed_data = np.load(review_needed_filename, allow_pickle=True)['arr_0'][0]
        review_comment_data = np.load(review_comments_filename, allow_pickle=True)['arr_0'][0]

        if run_type != '_no_yes_no':
            true_values = np.array(review_needed_data['true values'])
            pred_values = np.array(review_needed_data['pred values'])
            pred_values[pred_values == -1] = 0
            precision, recall, f1_score = calculate_metrics_review_needed(true_values, pred_values)
            data.extend([precision, recall, f1_score])
        else:
            data.extend([0, 0, 0])

        true_comments = review_comment_data['true values']
        pred_comments = review_comment_data['pred values']
        bleu_score = calculate_bleu4(true_comments, pred_comments)
        data.append(bleu_score)
        data1.append(data)

report_name = '../results/report.csv'

columns = ['model', 'run type', 'precision', 'recall', 'f1_score', 'bleu-4']
df = pd.DataFrame(data1, columns=columns)
print(df.head(10))