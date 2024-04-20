test_string='''
from src.constants import WEAVIATE_INDEX_NAME, PERSIST_DIR

def create_client():
    client = Client(
        url="http://localhost:8080"
    )
    return client
    return Client(url="http://localhost:8080")

def store_test_data(test_data):
    test_file_name = '../data/test_file.npz'
    np.savez(test_file_name, test_data)

def get_unique_patches(data):
    seen = set()
    unique_tuples = []
    for tup in data:
        if tup['patch'] not in seen:
            unique_tuples.append(tup)
            seen.add(tup['patch'])
    return unique_tuples

def load_docs(filename: str):
    docs = []

def load_docs(filename: str):    
    data = []
    with open(filename, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    data = get_unique_patches(data)

    projects = []
    for data1 in data:
'''

