import pandas as pd
import numpy as np

train_data=pd.read_csv("../first_train_index.csv")
train_matrix=[]
for i in train_data.index:
    temp=open('../train_data/'+str(train_data.ix[i,'id'])+'.txt').read()
    train_matrix.append(list(eval(temp)))



