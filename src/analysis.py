import pandas as pd
import numpy as np

model_name = 'mistral'
review_needed_filename = '../results/review_comment_{}.npz'.format(model_name)

review_needed_data = np.load(review_needed_filename, allow_pickle=True)['arr_0'][0]

print(type(review_needed_data))
print(review_needed_data)

print(review_needed_data['Ids'])
print(review_needed_data['true values'])
print(review_needed_data['pred values'])

