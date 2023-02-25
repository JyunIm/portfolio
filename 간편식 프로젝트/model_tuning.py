import pandas as pd
import pickle
from category_encoders import TargetEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import model_train as mo

df = pd.read_csv('df.csv')

X_train_e, y_train, X_val_e, y_val, X_test_e, y_test = mo.data_split(df)

params = {'num_leaves' : [10, 20, 30, 40],
          'min_data_in_leaf' : [100, 500, 1000],
          'learning_rate' : [0.01, 0.02, 0.05, 0.1]}

clf = GridSearchCV(
    XGBRegressor(random_state=42),
    param_grid=params,
    cv = 3,
    scoring = 'neg_mean_squared_error',
    verbose = 1,
    n_jobs = -1
)

clf.fit(X_train_e, y_train)
print('최적 하이퍼파라미터 : ', clf.best_params_)

# 모델 선택 및 최종 결과 확인
model = LGBMRegressor(learning_rate=0.1, min_data_in_leaf=100, 
                      num_leaves=20, random_state=42)
model.fit(X_train_e, y_train)
y_pred = model.predict(X_val_e)
print("검증 데이터 성능 확인")
print('MAE : ', mean_absolute_error(y_val, y_pred).round(2))
print('MSE : ', mean_squared_error(y_val, y_pred).round(2))
print('RMSE : ', (mean_squared_error(y_val, y_pred)**0.5).round(2))
print('R2 : ', r2_score(y_val, y_pred).round(2))
y_pred = model.predict(X_test_e)
print('\n테스트 데이터 성능 확인')
print('MAE : ', mean_absolute_error(y_test, y_pred).round(2))
print('MSE : ', mean_squared_error(y_test, y_pred).round(2))
print('RMSE : ', (mean_squared_error(y_test, y_pred)**0.5).round(2))
print('R2 : ', r2_score(y_test, y_pred).round(2))


# 모델 부호화
with open('model.pkl', 'wb') as pickle_file :
    pickle.dump(model, pickle_file)