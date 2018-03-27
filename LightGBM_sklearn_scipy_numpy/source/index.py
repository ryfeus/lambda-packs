import lightgbm as lgb
import numpy

def handler(event, context):
    dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
    X = dataset[:, 0:8]
    Y = dataset[:, 8]

    # lgb_train = lgb.Dataset(X[:600,:], Y[:600])
    # lgb_eval = lgb.Dataset(X[600:,:], Y[600:], reference=lgb_train)
    # params = {
    #     'task': 'train',
    #     'boosting_type': 'gbdt',
    #     'objective': 'binary',
    #     'metric': {'l2', 'auc'},
    #     'num_leaves': 21,
    #     'learning_rate': 0.05,
    #     'feature_fraction': 0.9,
    #     'bagging_fraction': 0.8,
    #     'bagging_freq': 5,
    #     'verbose': 0
    # }

    # gbm = lgb.train(params,
    #                 lgb_train,
    #                 num_boost_round=50,
    #                 valid_sets=lgb_eval,
    #                 early_stopping_rounds=5)

    # gbm.save_model('model.txt')

    bst = lgb.Booster(model_file='model.txt')
    Ypred = bst.predict(X)
    print(numpy.mean((Ypred>0.5)==(Y==1)))
    return 0