import pandas as pd
import numpy as np
from pylab import *
import itertools
from sklearn.model_selection import KFold,RepeatedKFold
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

# df = pd.read_csv('raw_old.csv')
df = pd.read_excel('../raw.xlsx')
names = df.columns.to_list()
for i in ['Ele','Compunds','Eg','Nb','EN','FIE','MR',
         'AN','MP','BP','Bh','Ehull','SOA','Eads','delta_G']:
    names.remove(i)

df_norm = (df[names]-np.mean(df[names]))/np.std(df[names])

y = np.array(df_norm['Eads']).reshape(-1,1)
names.remove('Eads')
# x = np.array(df[names]).reshape(-1,len(names))

model = SVR(kernel='rbf', gamma='scale')

for num_features in range(1,6):

    columns = ['features','R2','R2_train','R2_test','RMSE_train',
               'MAE_train','RMSE_test','MAE_test','R2_train_max','R2_train_min',
               'R2_test_max','R2_test_min','RMSE_train_max','RMSE_train_min',
               'MAE_train_max','MAE_train_min','RMSE_test_max','RMSE_test_min',
               'MAE_test_max','MAE_test_min'
              ]

    features,R2,R2_train,R2_test,RMSE_train,MAE_train,RMSE_test,MAE_test=[],[],[],[],[],[],[],[]
    R2_train_max,R2_train_min,R2_test_max,R2_test_min,RMSE_train_max,RMSE_train_min=[],[],[],[],[],[]
    MAE_train_max,MAE_train_min,RMSE_test_max,RMSE_test_min,MAE_test_max,MAE_test_min=[],[],[],[],[],[]
    
    for i in itertools.combinations(names, num_features):
        x = np.array(df_norm[list(i)]).reshape(-1,num_features)
        features.append(i)
        folds,repeats = 10,50
        kf = RepeatedKFold(n_splits=folds, n_repeats=repeats, random_state=10)
        r2 = r2_score(y,model.fit(x,y.ravel()).predict(x))
        R2.append(r2)
        
        results_all = ['r2_train','r2_test','rmse_train',
                       'mae_train','rmse_test','mae_test']
        r2_train,r2_test,rmse_train,mae_train,rmse_test,mae_test=[],[],[],[],[],[]
        for train, test in kf.split(x,y):
            x_train,y_train = x[train],y[train]
            x_test,y_test = x[test],y[test]
            
            y_train_pred = model.fit(x_train,y_train.ravel()).predict(x_train)
            y_test_pred = model.fit(x_train,y_train.ravel()).predict(x_test)
            std = np.std(df['Eads']); mean = np.mean(df['Eads'])
            r2_train.append(r2_score(y_train*std+mean,y_train_pred*std+mean))
            r2_test.append(r2_score(y_test*std+mean,y_test_pred*std+mean))
            mae_test.append(mean_absolute_error(y_test*std+mean,y_test_pred*std+mean))
            mae_train.append(mean_absolute_error(y_train*std+mean,y_train_pred*std+mean))
            rmse_test.append(mean_squared_error(y_test*std+mean,y_test_pred*std+mean)**0.5)
            rmse_train.append(mean_squared_error(y_train*std+mean,y_train_pred*std+mean)**0.5)
        
        R2_train.append(np.array(r2_train).mean())
        R2_test.append(np.array(r2_test).mean())
        RMSE_train.append(np.array(rmse_train).mean())
        MAE_train.append(np.array(mae_train).mean())
        RMSE_test.append(np.array(rmse_test).mean())
        MAE_test.append(np.array(mae_test).mean())
        R2_train_max.append(np.array(r2_train).max())
        R2_train_min.append(np.array(r2_train).min())
        R2_test_max.append(np.array(r2_test).max())
        R2_test_min.append(np.array(r2_test).min())
        RMSE_train_max.append(np.array(rmse_train).max())
        RMSE_train_min.append(np.array(rmse_train).min())
        MAE_train_max.append(np.array(mae_train).max())
        MAE_train_min.append(np.array(mae_train).min())
        RMSE_test_max.append(np.array(rmse_test).max())
        RMSE_test_min.append(np.array(rmse_test).min())
        MAE_test_max.append(np.array(mae_test).max())
        MAE_test_min.append(np.array(mae_test).min())
        
        results = pd.DataFrame(data = list(zip(features,R2,R2_train,R2_test,RMSE_train,MAE_train,
                                         RMSE_test,MAE_test,R2_train_max,R2_train_min,R2_test_max,
                                               R2_test_min,RMSE_train_max,RMSE_train_min,
                                               MAE_train_max,MAE_train_min,RMSE_test_max,
                                               RMSE_test_min,MAE_test_max,MAE_test_min)),columns=columns)
        results = results.sort_values(by=['R2_test'],ascending=False)
    print('------Dimension '+str(num_features)+' over------')   
    results.to_csv('order'+str(num_features)+f'_svr.csv',index=False)
