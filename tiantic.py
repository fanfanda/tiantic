import pandas as pd

train_data=pd.read_csv("../first_train_index.csv")
train_data['data']=''

for i in train_data.index:
    temp=open('../train_data/'+str(train_data.ix[i,'id'])+'.txt').read()
    train_data.ix[i,'data']=eval(temp)

