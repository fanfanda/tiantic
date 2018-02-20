import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import preprocessing
from sklearn import metrics
from sklearn.cross_validation import train_test_split

train_data=pd.read_csv("../first_train_index.csv")
train_data['data']=''
# train_matrix=np.zeros((len(train_data),2600))
for i in train_data.index:
    temp=open('../train_data/'+str(train_data.loc[i,'id'])+'.txt').read()
    train_data.ix[i,'data']=np.array(eval(temp)).reshape(-1,1)
    # train_matrix[i]=np.array(eval(temp))

type_star=preprocessing.LabelEncoder()
type_star.fit(train_data.type)
train_data['label']=type_star.transform(train_data.type)

class_num=len(type_star.classes_)

# x_train, x_test, y_train, y_test = train_test_split(train_matrix, train_data.label, test_size = 0.2)

# test_data=pd.read_csv("../first_test_index.csv")
# test_matrix=np.zeros((len(test_data),2600))
# for i in test_data.index:
#     temp=open('../test_data/'+str(test_data.loc[i,'id'])+'.txt').read()
#     test_matrix[i]=np.array(eval(temp))
# dMtest=xgb.DMatrix(test_matrix)
# dMtrain=xgb.DMatrix(train_matrix,label=train_data.label)

# params={'booster':'gbtree',
#     	'objective': 'multi:softprob',
#         'num_class':class_num,
#         #'eta': 0.1,
#         # 'max_depth': 9,
#         'subsample':0.8,
#         'colsample_bytree':0.8,
#         'colsample_bylevel':0.8,
#     	  'nthread':12,
#         'silent':1
#     	}
# watchlist = [(dMtrain,'train')]
# model = xgb.train(params,dMtrain,num_boost_round=65,evals=watchlist)



# test_list=model_pri.predict(dataset2)

# test_list_index=list(map(lambda x:x.argsort()[-1:],test_list))
# metrics.f1_score(y_test, test_list_index, average='macro')

# dMtrain=xgb.DMatrix(x_test,label=y_test)
# dMtest=xgb.DMatrix(x_train,label=y_train)
# model_m1 = xgb.train(params,dMtrain,num_boost_round=65,evals=watchlist)