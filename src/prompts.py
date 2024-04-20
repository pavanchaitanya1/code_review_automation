
REVIEW_NEEDED_PROMPT_PATCH = '''
You are a lead software engineer performing code reviews.

For the below patch of code, answer whether it needs review or not. Example patches of code and expected answers are given below. Use these examples as reference context and generate response in the same format as the answers. The response should only be in json format.

Patch:

{}
'''

REVIEW_NEEDED_PROMPT_EXAMPLE = '''

Example {}:

Patch : 

{}

Answer :

{{"reviewNeeded" : "{}"}}
'''

REVIEW_COMMENT_PROMPT_PATCH = '''
You are a lead software engineer performing code reviews.

For the below patch of code, generate one review comment. Although there could be multiple issues with the patch of code, only giv the most important suggestion. Example patches of code and expected answers are given below. Use these examples as reference context and generate response in the same format as the answers. The response should only be in json format.

Patch:

{}
'''

REVIEW_COMMENT_PROMPT_EXAMPLE = '''

Example {}:

Patch : 

{}

Answer :

{{"reviewComment" : "{}"}}
'''