from src.code_reviewer import CodeReviewer
from src.util import load_test_data
import pandas as pd
import argparse

def save_review_needed(ids, true_results, pred_results, model_name):
    data = {'Ids': ids, 'true values': true_results, 'pred values': pred_results}
    df = pd.DataFrame(data)
    df.to_csv('../results/review_needed1_{}.csv'.format(model_name), index=False)

def save_review_comment(ids, true_results, pred_results, model_name):
    data = {'Ids': ids, 'true values': true_results, 'pred values': pred_results}
    df = pd.DataFrame(data)
    df.to_csv('../results/review_comment1_{}.csv'.format(model_name), index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='mistral')
    parser.add_argument('--use_ollama', type=bool, default='True')
    args = parser.parse_args()
    model_name = args.model_name
    use_ollama = args.use_ollama
    
    code_reviewer = CodeReviewer(use_ollama=use_ollama, model_name=model_name)
    test_data = load_test_data()

    # test_data = test_data[:1]

    true_y = []
    true_msg = []
    ids =[]
    for data in test_data:
        true_y.append(data.y)
        true_msg.append(data.msg)
        ids.append(data.id)
    
    pred_y = []
    pred_msg = []

    count = 1
    for data in test_data:
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
        review_comment = code_reviewer.generate_review_comment(patch)
        print(review_comment)
        pred_msg.append(review_comment)
        count += 1
    
    save_review_needed(ids, true_y, pred_y, model_name)
    save_review_comment(ids, true_msg, pred_msg, model_name)

if __name__ == '__main__':
    main()
