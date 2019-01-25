import numpy as np
from scipy.special import expit

INPUT_SIZE = 2
OUTPUT_SIZE = 1
BATCH_SIZE = 4
TRIAL_SIZE = 100
PICK_SIZE = TRIAL_SIZE // 10
RUNS = 10


def make_model(input_size, trial_size, output_size, max_w=1.):
    model = {}

    W = np.random.uniform(-max_w, max_w, size=[trial_size, input_size, output_size])
    model["W"] = W

    b = np.random.uniform(-max_w,max_w,size=[trial_size, output_size])
    model["b"] = b

    return model


def update_model(model, new_weights):
    model["W"] = new_weights[:,0:2].reshape(TRIAL_SIZE,INPUT_SIZE,OUTPUT_SIZE)

    model["b"] = new_weights[:,-1].reshape(TRIAL_SIZE,OUTPUT_SIZE)


def forward_pass(model,x):
    n = np.transpose(np.matmul(x, model["W"]),axes=(1,0,2)) + model["b"]
    return expit(n)


def calc_mse(y, y_pred):
    err = np.transpose(y,axes=(1,0,2)) - y_pred
    return np.mean(err * err,axis=1)


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

    return average_weights, cov_weights


def main():
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]],dtype=np.float32)

    Y = np.array([[0],
                  [1],
                  [1],
                  [1]],dtype=np.float32)

    model = make_model(INPUT_SIZE,TRIAL_SIZE, OUTPUT_SIZE)

    last_err = 1.

    for i in range(RUNS):
        o = forward_pass(model, X)
        err = calc_mse(o,Y)
        min_err = np.min(err)
        print("Run: {} Min MSE: {} Change: {}".format(i,min_err, last_err - min_err))
        last_err = min_err

        best_models = np.argpartition(err.ravel(),PICK_SIZE)[:PICK_SIZE]

        avg_weights, cov_weights = update_distribution(model, best_models)

        new_weights = np.random.multivariate_normal(avg_weights,cov_weights,TRIAL_SIZE)

        update_model(model, new_weights)

    o = forward_pass(model, X)
    print(o)
    print(avg_weights)
    print(cov_weights)


if __name__ == "__main__":
    main()
