import numpy as np
from scipy.special import expit

INPUT_SIZE = 2
HIDDEN_SIZE = 3
OUTPUT_SIZE = 1
BATCH_SIZE = 4
TRIAL_SIZE = 2000
PICK_SIZE = TRIAL_SIZE // 3
GROW_SIZE = TRIAL_SIZE - PICK_SIZE
RUNS = 10


def make_model(input_size, trial_size, output_size, max_w=3.):
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
    model["W1"] = new_weights[:,0:6].reshape(TRIAL_SIZE,INPUT_SIZE,HIDDEN_SIZE)
    model["b1"] = new_weights[:,6:9].reshape(TRIAL_SIZE,HIDDEN_SIZE)
    model["W2"] = new_weights[:,9:12].reshape(TRIAL_SIZE,HIDDEN_SIZE,OUTPUT_SIZE)
    model["b2"] = new_weights[:,12].reshape(TRIAL_SIZE,OUTPUT_SIZE)


def forward_pass(model,x):
    """
    Uses SIGN activation function
    :param model: collection of parameters
    :param x:
    :return:
    """
    n1 = np.transpose(np.matmul(x, model["W1"]),axes=(1,0,2)) + model["b1"]
    # o1 = expit(n1)
    o1 = np.where(n1 > 0, 1., 0.)

    n2 = np.transpose(np.matmul(np.transpose(o1,axes=(1,0,2)),
                                model["W2"]),axes=(1,0,2)) + model["b2"]
    return np.where(n2 > 0, 1., 0.)
    # expit(n2)


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

    return average_weights, cov_weights, x


def extract_best_model(model, best_):
    best_model = {"W1": model["W1"][best_],
                  "b1": model["b1"][best_],
                  "W2": model["W2"][best_],
                  "b2": model["b2"][best_]}

    return best_model


def main():
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]],dtype=np.float32)

    Y = np.array([[0],
                  [1],
                  [1],
                  [0]],dtype=np.float32)

    model = make_model(INPUT_SIZE,TRIAL_SIZE, OUTPUT_SIZE)

    last_err = 1.

    for i in range(RUNS):
        o = forward_pass(model, X)
        err = calc_mse(o,Y)
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

    best_o = forward_pass(best_model, X)
    print(best_o)

if __name__ == "__main__":
    main()
