import numpy as np
import pandas as pd
import os
import pickle
import gc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

avgfp = pd.read_csv("avgfp.tsv", sep = "\t")
input_data = np.empty((20,20)) #change input if stacked with reverse
feat_ids = avgfp.variant[:19900].to_numpy()
feat_ids = feat_ids+".npy"

r = 0
train_index = []
for i in feat_ids:
    #data = np.load('tmp.npy')
    try:
        new = np.load("./20K_feat_extract/"+i)

        if r == 0:
            input_data = np.stack([input_data,new], axis = 0)
            r+=1
        else:
            new = np.expand_dims(new,axis = 0)
            #print(new.shape)
            input_data = np.concatenate([input_data,new])

       # process data
        train_index.append(i)
        del new
        gc.collect()
    except:
        pass
    
input_data = input_data[1:,]
np.save("./pssm_features.npy", input_data)

with open("./train_index.pickle", "wb") as f:
    pickle.dump(train_index,f)