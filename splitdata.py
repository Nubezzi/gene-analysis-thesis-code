from core import *
import itertools
import random
import pandas as pd
import numpy as np
import sys
import os

filename = sys.argv[1]
out_dir = sys.argv[2]
    
data   = pd.read_csv(filename) 

dfs = np.array_split(data, len(data) // 500 + 1)

for i in range(len(dfs)):
    x = dfs[i]
    
    if not os.path.isdir(f"{out_dir}"):
        os.makedirs(f"{out_dir}")
    
    x.to_csv(f"{out_dir}/data{i}.csv")