import pandas as pd
from sklearn import preprocessing
import xgboost as xgb

train = pd.read_csv('/Users/Karel/Documents/Kaggle/Sberbank/train.csv', parse_dates=['timestamp'])
test = pd.read_csv('/Users/Karel/Documents/Kaggle/Sberbank/test.csv', parse_dates=['timestamp'])
macro = pd.read_csv('/Users/Karel/Documents/Kaggle/Sberbank/macro.csv', parse_dates=['timestamp']
                    ,usecols=["timestamp"])
#                       usecols=["timestamp","balance_trade", "balance_trade_growth", "eurrub", "average_provision_of_build_contract",
#"micex_rgbi_tr", "micex_cbi_tr", "deposits_rate", "mortgage_value", "mortgage_rate",
#"income_per_cap", "rent_price_4+room_bus", "museum_visitis_per_100_cap", "apartment_build"])

id_test = test.id
train = pd.merge_ordered(train, macro, on='timestamp', how='left')
y_train = train["price_doc"]
x_train = train.drop(["id", "timestamp", "price_doc"], axis=1)
x_test = test.drop(["id", "timestamp"], axis=1)


for c in x_train.columns:
    if x_train[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_train[c].values))
        x_train[c] = lbl.transform(list(x_train[c].values))
        # x_train.drop(c,axis=1,inplace=True)

for c in x_test.columns:
    if x_test[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_test[c].values))
        x_test[c] = lbl.transform(list(x_test[c].values))
        # x_test.drop(c,axis=1,inplace=True)


dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)

xgb_params = {
    'eta': 0.01,                    #0.02 -> 0.01->0.01 -> 0.01 -> 0.05 -> 0.02 -> 0.01 -> 0.01
    'max_depth': 6,                 #3    -> 5   -> 6   -> 6    -> 6    -> 6    -> 7    -> 7
    'subsample': 0.8,               #0.7  -> 0.7 -> 0.7 -> 0.8  -> 0.8  -> 0.8  -> 0.8  -> 0.7
    'colsample_bytree': 0.8,        #0.7  -> 0.7 -> 0.7 -> 0.8  -> 0.8  -> 0.6  -> 0.8  -> 0.7
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'silent': 1,
    'min_child_weight':1
}

cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=5000, early_stopping_rounds=20,
    verbose_eval=50, show_stdv=False)

num_boost_rounds = len(cv_output)
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round= num_boost_rounds)

y_predict = model.predict(dtest)
output = pd.DataFrame({'id': id_test, 'price_doc': y_predict})
output.to_csv('Sub_3.csv', index=False)


#train-rmse:1.77955e+06	test-rmse:2.61139e+06
#train-rmse:1.59941e+06	test-rmse:2.60447e+06

#[1650]	train-rmse:1.50633e+06	test-rmse:2.59532e+06 -> fail 32200?
#[1400]	train-rmse:1.62385e+06	test-rmse:2.60716e+06 -> best...0.31525
#[350]	train-rmse:1.54295e+06	test-rmse:2.62358e+06
#[750]	train-rmse:1.60104e+06	test-rmse:2.60934e+06
#[1150]	train-rmse:1.46332e+06	test-rmse:2.61324e+06 -> 0.31536
#[1350]	train-rmse:1.42575e+06	test-rmse:2.61186e+06 -> 0.31628