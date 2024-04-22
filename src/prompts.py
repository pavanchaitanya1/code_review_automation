
REVIEW_NEEDED_PROMPT_PATCH = '''
You are a lead software engineer performing code reviews.

For the below patch of code, answer whether it needs review or not. Example patches of code and expected answers are given above. Use these examples as reference context and generate response in the same format as the answers. The response should only be in json format.

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

REVIEW_NEEDED_PROMPT_NO_RAG = '''
You are a lead software engineer performing code reviews.

For the below patch of code, answer whether it needs review or not. The response should only be in json format like below.

{{"reviewNeeded" : "true" }} or {{"reviewNeeded" : "false" }}

Patch:

{}
'''

REVIEW_COMMENT_PROMPT_PATCH = '''
You are a lead software engineer performing code reviews.

For the below patch of code, generate one review comment. Although there could be multiple issues with the patch of code, only give the most important suggestion. Example patches of code and expected answers are given above. Use these examples as reference context and generate response in the same format as the answers. The response should only be in json format.

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

REVIEW_COMMENT_PROMPT_NO_RAG = '''
You are a lead software engineer performing code reviews.

For the below patch of code, generate one review comment. Although there could be multiple issues with the patch of code, only give the most important suggestion. The response should only be in json format like below.

{{"reviewComment" : "" }}

Patch:

{}
'''


REVIEW_COMMENT_PROMPT_NO_YES_NO = '''
You are a lead software engineer performing code reviews.

For the below patch of code, generate one review comment. Although there could be multiple issues with the patch of code, only give the most important suggestion. The response should only be in json format like below. If you think no review is needed for this patch of code, give the reviewComment as empty string.

{{"reviewComment" : "" }}

Patch:

{}
'''

REVIEW_COMMENT_PROMPT_GITHUB = '''
You are a lead software engineer performing code reviews.

For the below patch of code, generate one review comment. Although there could be multiple issues with the patch of code, only give the most important suggestion. Example patches of code and expected answers are given above. Use these examples as reference context and generate response in the same format as the answers. The response should only be in json format:
{{"reviewComment" : " ", "lineNumber": " "}}


Patch:

{}
'''