from src.code_reviewer import CodeReviewer
from src.util import load_test_data
import pandas as pd


def save_predictions(ids, true_results, pred_results, model_name):
    data = {'Ids': ids, 'true values': true_results, 'pred values': pred_results}
    df = pd.DataFrame(data)
    df.to_csv('../results/review_needed_{}.csv'.format(model_name), index=False)

def main():
    model_name = 'llama2'
    code_reviewer = CodeReviewer(use_ollama=True, model_name=model_name)
    test_data = load_test_data()

    true_values = []
    ids =[]
    for data in test_data:
        true_values.append(data.y)
        ids.append(data.id)
    
    pred_values = []

    count = 0
    for data in test_data:
        patch = data.patch
        id = data.id
        try:
            review_needed = code_reviewer.is_review_needed(patch)
            print(count, id, review_needed)
            if review_needed:
                pred_values.append(1)
            else:
                pred_values.append(0)
        except Exception as ex:
            pred_values.append(-1)
            print(count, id)
            print(ex)
        count += 1
    
    save_predictions(ids, true_values, pred_values, model_name)

if __name__ == '__main__':
    main()
