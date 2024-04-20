from src.code_reviewer import CodeReviewer
from src.util import load_test_data

def main():
    code_reviewer = CodeReviewer()
    test_data = load_test_data()

    test_data = test_data[:1]

    for data in test_data:
        patch = data.patch
        review_needed = code_reviewer.is_review_needed(patch)
        print(review_needed)
        if review_needed:
            review_comment = code_reviewer.generate_review_comment(patch)
            print(review_comment)

if __name__ == '__main__':
    main()

