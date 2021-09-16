import numpy as np
import pandas as pd
import re
import glob
import os
import sys

ref = "SKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
avgfp = pd.read_csv("avgfp.tsv", sep = "\t")
aas = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
         "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]
def transform_pssm_400(filepath):
    pssm = np.loadtxt(filepath, skiprows=3,usecols=np.arange(2,22), max_rows=238)
    seq = np.genfromtxt(filepath, skip_header=3, usecols =[1],max_rows = 237, dtype='str')
    pssm_transformed = np.zeros((20,20))
    for i in range(20):
        residues = np.where(seq==aas[i])[0]
        pssm_transformed[i] = pssm[residues,:].sum(axis = 0)
    max_sc = np.max(pssm_transformed)
    min_sc = np.min(pssm_transformed)
    pssm_transformed = (pssm_transformed-min_sc)/(max_sc-min_sc)
    return pssm_transformed
        
    
all_fp = glob.glob("./20K_out_pssm/*")
for i in range(len(all_fp)):
    fname = os.path.basename(all_fp[i]).split(".")[0]
    trans_pssm = transform_pssm_400(all_fp[i])
    np.save("./20K_feat_extract/"+fname+".npy",trans_pssm)
        
