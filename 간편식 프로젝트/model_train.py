### ML 모델 ###
# 라이브러리 불러오기
import pandas as pd
import numpy as np
import os
import re
import psycopg2
import warnings
from dotenv import load_dotenv
from category_encoders import TargetEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
warnings.filterwarnings('ignore')

# 환경변수 가져오기
load_dotenv()
psycopg_host = os.getenv('ELEPHANTSQL_HOST')
psycopg_db = os.getenv('ELEPHANTSQL_DB')
psycopg_user = os.getenv('ELEPHANTSQL_USER')
psycopg_password = os.getenv('ELEPHANTSQL_PASSWORD')

# DB에서 데이터 불러오기
conn = psycopg2.connect(
    host = psycopg_host,
    database = psycopg_db,
    user = psycopg_user,
    password = psycopg_password
)

cur = conn.cursor()
cur.execute('SELECT * FROM public.naver;')
naver = cur.fetchall()

naver = np.array(naver)
naver = pd.DataFrame(data=naver, columns = ['id','name','price','delivery_fee','category', 'etc'])

# EDA 진행
def eda(df):
    df1 = df.copy()
    df1['review'] = None
    df1['sales'] = None
    df1['price'] = df1['price'].str.replace(',','').str.extract(r'(\d+)')
    df1['delivery_fee'] = df1['delivery_fee'].str.replace(',','').str.extract(r'(\d+)')
    for i in range(len(df1)):
        df1['category'][i] = df1['category'][i].replace('식품밀키트','')
        df1['review'][i] = df1['etc'][i].split(' ')[0]
        df1['sales'][i] = df1['etc'][i].split(' ')[-1]
        if df1['sales'][i].find('구매건수'):
            df1['sales'][i] = 0
        else:
            pass
    df1['review'] = df1['review'].str.replace(',','').str.extract(r'(\d+)')
    df1['sales'] = df1['sales'].str.replace(',','').str.extract(r'(\d+)')
    df1.drop(columns='etc', inplace=True)
    df1[['price','delivery_fee','review','sales']] = df1[['price','delivery_fee','review','sales']].astype(float)
    df1[['delivery_fee','review','sales']] = df1[['delivery_fee','review','sales']].fillna(0)
    df1.category = df1.category.map({'찌개/국':1, '간식/디저트':2, '볶음/튀김':3,'구이':4,'면/파스타':5})
    df1['category'] = df1['category'].astype(int)
    p = re.compile(r'[^ 가-힣]+')
    for i in range(df1.shape[0]):
       df1['name'][i] = re.sub(p, '', df1['name'][i]).split(' ')
    df1['sales'] = df1['sales']/max(df1['sales'])
    return df1

# 불용어 사전 생성
def make_stopwords(df):
    data = df.copy()
    data = data['name'].apply(lambda x : pd.Series(x)).stack().reset_index(1, name='name').drop('level_1', axis=1)
    data_word = data['name'].value_counts()
    data_word = pd.DataFrame(data_word)
    if (data_word.quantile(0.95) / df.shape[0])[0] < 0.01 :
        value = data_word.quantile(0.95)[0]
        data_word = data_word.query(f'name < {value}')
        data_word = list(data_word.index)
    stopword = data_word + ['', '인분', '인', '밀키트']
    return stopword

# name column에서 불용어 제거 & 불용어가 제거된 keyword를 이어붙임
def del_stopwords(df, stop) :
    for i in range(df.shape[0]):
        df['name'][i] = [word for word in df['name'][i] if word not in stop]
        df['name'][i] = ','.join(df['name'][i])
    temp = df['name'].str.split(',', expand=True).fillna(0)
    df = pd.concat([df,temp], axis=1).drop('name', axis=1)
    return df

# 데이터를 train, val, test로 나누는 함수
def data_split(df):
    encoding = TargetEncoder()
    target = 'sales'
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    train, val = train_test_split(train, test_size=0.2, random_state=42)
    X_train = train.drop(target, axis=1)
    X_val = val.drop(target, axis=1)
    X_test = test.drop(target, axis=1)
    y_train = train[target]
    y_val = val[target]
    y_test = test[target]
    X_train_enc = encoding.fit_transform(X_train, y_train)
    X_val_enc = encoding.transform(X_val)
    X_test_enc = encoding.transform(X_test)
    return X_train_enc, y_train, X_val_enc, y_val, X_test_enc, y_test

# 모델을 학습시키는 함수
def model_train(X_t, y_t, X_v, y_v, model):
    X_tr = X_t.copy()
    y_tr = y_t.copy()
    X_va = X_v.copy()
    y_va = y_v.copy()
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_va)
    print('MAE : ', mean_absolute_error(y_va, y_pred).round(2))
    print('MSE : ', mean_squared_error(y_va, y_pred).round(2))
    print('RMSE : ', (mean_squared_error(y_va, y_pred)**0.5).round(4))
    print('R2 : ', r2_score(y_va, y_pred).round(2))

na = eda(naver)
stopwords = make_stopwords(na)
na = del_stopwords(na, stopwords)
X_train_e, y_train, X_val_e, y_val, X_test_e, y_test = data_split(na)

rf_model = RandomForestRegressor(random_state=42)
dt_model = DecisionTreeRegressor(random_state=42)
boo_model = XGBRegressor(eval_metric = 'rmse',random_state=42)
light_model = LGBMRegressor(eval_metric = 'rmse',random_state=42)

print('\nRandomForest')
model_train(X_train_e, y_train, X_val_e, y_val, rf_model)
print('\nDecisionTree')
model_train(X_train_e, y_train, X_val_e, y_val, dt_model)
print('\nXGB')
model_train(X_train_e, y_train, X_val_e, y_val, boo_model)
print('\nLGBM')
model_train(X_train_e, y_train, X_val_e, y_val, light_model)

na.to_csv('df.csv')