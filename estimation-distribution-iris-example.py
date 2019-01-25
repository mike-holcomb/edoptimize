'''
Mike Holcomb
Example of training network with one layer MLP on Iris data set
'''
import numpy as np
import pandas as pd
from scipy.special import expit
import sklearn

INPUT_SIZE = 4
HIDDEN_SIZE = 10
OUTPUT_SIZE = 3
BATCH_SIZE = 60
TRIAL_SIZE = 500
PICK_SIZE = TRIAL_SIZE // 20
GROW_SIZE = TRIAL_SIZE - PICK_SIZE
RUNS = 20


def softmax(x):
    bs = x.shape[0]
    ts = x.shape[1]
    res = np.exp(x) / np.repeat(np.sum(np.exp(x),axis=2).reshape(bs,ts,1),3,axis=2)
    # print(res[0,0,:])
    return res


def make_model(input_size, trial_size, output_size, max_w=2.):
    """
    Make MLP with one hidden layer
    :param input_size: number of inputs
    :param trial_size: number of concurrent nets to evaluate
    :param output_size: number of outputs
    :param max_w: initial weight distribution
    :return:
    """
    model = {}

    hidden_size = HIDDEN_SIZE

    # Input to Hidden Layer
    W1 = np.random.uniform(-max_w, max_w, size=[trial_size, input_size, hidden_size])
    model["W1"] = W1

    b1 = np.random.uniform(-max_w,max_w,size=[trial_size, hidden_size])
    model["b1"] = b1

    # Hidden Layer to Output
    W2 = np.random.uniform(-max_w, max_w, size=[trial_size, hidden_size, output_size])
    model["W2"] = W2

    b2 = np.random.uniform(-max_w,max_w,size=[trial_size, output_size])
    model["b2"] = b2

    return model


def update_model(model, new_weights):
    model["W1"] = new_weights[:,0:40].reshape(TRIAL_SIZE,INPUT_SIZE,HIDDEN_SIZE)
    model["b1"] = new_weights[:,40:50].reshape(TRIAL_SIZE,HIDDEN_SIZE)
    model["W2"] = new_weights[:,50:80].reshape(TRIAL_SIZE,HIDDEN_SIZE,OUTPUT_SIZE)
    model["b2"] = new_weights[:,80:].reshape(TRIAL_SIZE,OUTPUT_SIZE)


def forward_pass(model,x):
    """
    Uses SIGN activation function
    :param model: collection of parameters
    :param x:
    :return:
    """
    n1 = np.transpose(np.matmul(x, model["W1"]),axes=(1,0,2)) + model["b1"]
    o1 = expit(n1)
    # o1 = np.where(n1 > 0, 1., 0.)

    n2 = np.transpose(np.matmul(np.transpose(o1,axes=(1,0,2)),
                                model["W2"]),axes=(1,0,2)) + model["b2"]
    return softmax(n2)
    # np.where(n2 > 0, 1., 0.)
    # expit(n2)


def calc_mse(y, y_pred):
    err = np.transpose(y,axes=(1,0,2)) - y_pred
    err = np.sum(err * err,axis=2)
    return np.mean(err,axis=1)


def update_distribution(model, best_models):
    x = None
    n = len(best_models)
    for weights in model.values():
        if x is None:
            x = weights[best_models].reshape(n,-1)
        else:
            x = np.concatenate((x,weights[best_models].reshape(n,-1)),axis=1)

    average_weights = np.mean(x,axis=0)
    cov_weights = np.cov(x,rowvar=False)

    return average_weights, cov_weights, x


def extract_best_model(model, best_):
    best_model = {"W1": model["W1"][best_].reshape(1,INPUT_SIZE,HIDDEN_SIZE),
                  "b1": model["b1"][best_].reshape(1,HIDDEN_SIZE),
                  "W2": model["W2"][best_].reshape(1,HIDDEN_SIZE,OUTPUT_SIZE),
                  "b2": model["b2"][best_].reshape(1,OUTPUT_SIZE)}

    return best_model


def load_data():
    data = pd.read_csv('iris.csv')
    np.random.shuffle(data.values)
    y = pd.get_dummies(data["species"]).values
    X = data.drop(["species"],axis=1).values
    x_mean = np.mean(X,axis=0)
    x_stdev = np.std(X,axis=0)
    X = (X - x_mean) / (x_stdev + 0.01)

    train_mask = np.isin(np.arange(150),
                         np.random.choice(150, 120, replace=False))

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[~train_mask], y[~train_mask]

    return (X_train, y_train, X_test,y_test)


def train(X, Y):
    model = make_model(INPUT_SIZE,TRIAL_SIZE, OUTPUT_SIZE)

    last_err = 1.

    for i in range(RUNS):
        shift = i % 2
        if shift == 0:
            shuffled = np.arange(120)
            np.random.shuffle(shuffled)
            X = X[shuffled]
            Y = Y[shuffled]
        o = forward_pass(model, X[shift*BATCH_SIZE:(shift+1)*BATCH_SIZE])
        err = calc_mse(o,Y[shift*BATCH_SIZE:(shift+1)*BATCH_SIZE])
        min_err = np.min(err)
        print("Run: {} Min MSE: {} Change: {}".format(i,min_err, last_err - min_err))
        last_err = min_err

        best_models = np.argpartition(err.ravel(),PICK_SIZE)[:PICK_SIZE]

        avg_weights, cov_weights, old_weights = update_distribution(model, best_models)

        new_weights = np.random.multivariate_normal(avg_weights,cov_weights,GROW_SIZE)
        new_weights = np.concatenate((new_weights,old_weights),axis=0)

        update_model(model, new_weights)

    o = forward_pass(model, X)
    err = calc_mse(o, Y)
    best_index = np.argmin(err,axis=0)
    best_model = extract_best_model(model,best_index)
    # print(best_model)

    return best_model


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data()
    print("Iris Classification - Estimation of Distribution Optimizer")
    print("Train set - class proportions: \t", np.mean(y_train,axis=0))
    print("Test set - class proportions: \t", np.mean(y_test, axis=0))

    best_model = train(X_train, y_train)
    o = forward_pass(best_model, X_test)
    err = calc_mse(o, y_test)
    min_err = np.min(err)
    print("TEST MSE: {} ".format(min_err))
    # print(o.shape)
    y_cat = np.argmax(o,axis=2).ravel()
    y_test_cat = np.argmax(y_test,axis=1).ravel()

    # print(y_cat)
    # print(y_test_cat)

    print(sklearn.metrics.classification_report(y_test_cat,y_cat))
