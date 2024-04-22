from flask import Flask, request
from src.code_reviewer import CodeReviewer
import os
import argparse

from github import Github

YOUR_GITHUB_TOKEN = os.environ.get('ASE_GITHUB_API_KEY')
REPO_NAME = "AI-SWEngineering/SWE1"

g = Github(YOUR_GITHUB_TOKEN)
repo = g.get_repo(REPO_NAME)

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='mistral')
args = parser.parse_args()
model_name = args.model_name

code_reviewer = CodeReviewer(model_name=model_name, top_k=2, github_bot=True)

app = Flask(__name__)

def fix_line_number(patch, line_number):
    line_number = int(line_number)
    lines = patch.strip().split('\n')
    start_line_num = int(lines[0].split(' ')[1].split(',')[0])
    end_line_num = start_line_num + len(lines) - 1
    if (not (line_number>= start_line_num and line_number<=end_line_num)):
        line_number = end_line_num
    return line_number


def execute_post(data):
    pull_re_no = int(data['pull_req_number'])
    pull_request = repo.get_pull(number= pull_re_no)
    diff = pull_request.get_files()
    commit = repo.get_commit(sha=pull_request.head.sha)

    for file in diff:
        filename = file.filename
        patch = file.patch
        
        review_needed = code_reviewer.is_review_needed(patch)
        print('Is Review Needed:', review_needed)

        if review_needed:
            review_comment, line_number = code_reviewer.generate_review_comment(patch)
            line_number = fix_line_number(patch, line_number)
            
            print(review_comment)
            print('Line Number: ', line_number)

            pull_request.create_review_comment(
                body = review_comment, 
                commit = commit, 
                path = filename, 
                line = line_number)

@app.route('/', methods=['GET', 'POST'])
def handle_request():
    if request.method == 'POST':
        data = request.get_json()
        execute_post(data)
        return 'done'

    elif request.method == 'GET':
        print("Received GET request data:", request.args)
        return "GET request received successfully!"

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=5000, debug=False)
