from src.code_reviewer import CodeReviewer
from src.util import load_test_data
import numpy as np
import argparse

def save_review_needed(ids, true_results, pred_results, model_name, is_rag):
    if is_rag:
        filename = '../results/review_needed_{}.npz'
    else:
        filename = '../results/review_needed_{}_no_rag.npz'
    data = [{'Ids': ids, 'true values': true_results, 'pred values': pred_results}]
    np.savez(filename.format(model_name), data)

def save_review_comment(ids, true_results, pred_results, model_name, is_rag):
    if is_rag:
        filename = '../results/review_comment_{}.npz'
    else:
        filename = '../results/review_comment_{}_no_rag.npz'
    data = [{'Ids': ids, 'true values': true_results, 'pred values': pred_results}]
    np.savez(filename.format(model_name), data)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='mistral')
    parser.add_argument('--use_rag', type=str, default='True')
    args = parser.parse_args()
    model_name = args.model_name
    use_rag = args.use_rag
    if use_rag == 'True':
        use_rag = True
    else:
        use_rag = False
    use_ollama = False
    
    code_reviewer = CodeReviewer(use_ollama=use_ollama, model_name=model_name, use_rag=use_rag)
    test_data = load_test_data()

    test_data = test_data[:1]

    true_y = []
    true_msg = []
    ids =[]
    for data in test_data:
        true_y.append(data.y)
        true_msg.append(data.msg)
        ids.append(data.id)
    
    pred_y = []
    pred_msg = []

    count = 0
    for data in test_data:
        count += 1
        patch = data.patch
        id = data.id
        print(count, id, '---------------')
        try:
            review_needed = code_reviewer.is_review_needed(patch)
            print(review_needed)
            if review_needed:
                pred_y.append(1)
            else:
                pred_y.append(0)
                pred_msg.append('')
                continue
        except Exception as ex:
            pred_y.append(-1)
            pred_msg.append('')
            print(ex)
            continue
        try:
            review_comment = code_reviewer.generate_review_comment(patch)
            print(review_comment)
            pred_msg.append(review_comment)
        except Exception as ex:
            print(ex)
            pred_msg.append('')

        
    
    save_review_needed(ids, true_y, pred_y, model_name, use_rag)
    save_review_comment(ids, true_msg, pred_msg, model_name, use_rag)

if __name__ == '__main__':
    main()
