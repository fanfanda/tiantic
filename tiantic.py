import pandas as pd
import numpy as np

train_data=pd.read_csv("../first_train_index.csv")
train_matrix=np.zeros((len(train_data),2600))
for i in train_data.index:
    temp=open('../train_data/'+str(train_data.loc[i,'id'])+'.txt').read()
    train_matrix[i]=np.array(eval(temp))



