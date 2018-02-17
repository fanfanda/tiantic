import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import preprocessing

train_data=pd.read_csv("../first_train_index.csv")
train_matrix=np.zeros((len(train_data),2600))
for i in train_data.index:
    temp=open('../train_data/'+str(train_data.loc[i,'id'])+'.txt').read()
    train_matrix[i]=np.array(eval(temp))

type_star=preprocessing.LabelEncoder()
type_star.fit(train_data.type)
train_data['label']=type_star.transform(train_data.type)

class_num=len(type_star.classes_)

dMtrain=xgb.DMatrix(train_matrix,label=train_data.label)
params={'booster':'gbtree',
    	'objective': 'multi:softprob',
        'num_class':class_num,
        #'eta': 0.1,
        # 'max_depth': 9,
        'subsample':0.8,
        'colsample_bytree':0.8,
        'colsample_bylevel':0.8,
    	'nthread':12,
        'silent':1
    	}
watchlist = [(dMtrain,'train')]
model = xgb.train(params,dMtrain,num_boost_round=65,evals=watchlist)


