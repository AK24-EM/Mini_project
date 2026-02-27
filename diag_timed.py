import pickle
import numpy as np
import threading
import os
import sys

def worker():
    try:
        with open('model/Kmean.pkl', 'rb') as f:
            model = pickle.load(f)
        print('CENTROIDS:')
        print(model.cluster_centers_)
        print('DONE')
    except Exception as e:
        print('ERROR:', e)

t = threading.Thread(target=worker)
t.start()
t.join(timeout=15)
if t.is_alive():
    print('TIMEOUT')
    os._exit(1)
