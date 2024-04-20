from src.code_reviewer import CodeReviewer
from src.util import load_test_data
import pandas as pd


def save_predictions(ids, true_results, pred_results):
    data = {'Ids': ids, 'true values': true_results, 'pred values': pred_results}
    df = pd.DataFrame(data)
    df.to_csv('../results/review_needed.csv', index=False)

def main():
    code_reviewer = CodeReviewer()
    test_data = load_test_data()

    print(len(test_data))

    true_values = []
    ids =[]
    for data in test_data:
        true_values.append(data.y)
        ids.append(data.id)

    pred_values = []

    for data in test_data:
        patch = data.patch
        review_needed = code_reviewer.is_review_needed(patch)
        if review_needed:
            pred_values.append(1)
        else:
            pred_values.append(0)
    
    save_predictions(ids, true_values, pred_values)

if __name__ == '__main__':
    main()
