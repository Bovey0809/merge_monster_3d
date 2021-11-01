import pickle

with open('./cuda_error_batch', 'rb') as f:
    data = pickle.load(f)

print(data)