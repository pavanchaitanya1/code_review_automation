from flask import Flask, request
from src.code_reviewer import CodeReviewer
import os

YOUR_GITHUB_TOKEN = os.environ.get('ASE_GITHUB_API_KEY')

from github import Github

app = Flask(__name__)

code_reviewer = CodeReviewer(model_name='mistral', top_k=2, use_ollama=True)

@app.route('/', methods=['GET', 'POST']) # type: ignore
def handle_request():
    if request.method == 'POST':
        data = request.get_json()  # Assuming the request is JSON data
        print("Received POST request data:", data)
        print("POST request received successfully!") 

        g = Github(YOUR_GITHUB_TOKEN)
        repo = g.get_repo("AI-SWEngineering/SWE1")
        pull_re_no = int(data['pull_req_number'])
        print(pull_re_no)
        pull_request = repo.get_pull(number= pull_re_no)  # Replace 1 with the pull request number
        diff = pull_request.get_files()


        # Print the diff
        for file in diff:
            #print(f"File: {file.filename}\nChanges:\n{file.patch}\n")
            review_needed = code_reviewer.is_review_needed(file.patch)
            print(review_needed)
            if review_needed:
                review_comment = code_reviewer.generate_review_comment(file.patch)
                print(review_comment)
        return 'done'

    elif request.method == 'GET':
        print("Received GET request data:", request.args)
        return "GET request received successfully!"

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=5000, debug=False)
