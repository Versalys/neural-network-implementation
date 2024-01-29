"""Main script for the solution."""

import numpy as np
import pandas as pd
import argparse

from matplotlib import pyplot as plt
from tqdm import tqdm

from npnn.model import Sequential
from npnn.modules import Dense, ELU, SoftmaxCrossEntropy, Flatten
from npnn.optimizer import SGD, Adam
import npnn


def _get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lr", help="learning rate", type=float, default=0.1)
    p.add_argument("--opt", help="optimizer", default="SGD")
    p.add_argument(
        "--epochs", help="number of epochs to train", type=int, default=20)
    p.add_argument(
        "--save_stats", help="Save statistics to file", action="store_true")
    p.add_argument(
        "--save_pred", help="Save predictions to file", action="store_true")
    p.add_argument("--dataset", help="Dataset file", default="mnist.npz")
    p.add_argument(
        "--test_dataset", help="Dataset file (test set)",
        default="mnist_test.npz")
    p.set_defaults(save_stats=False, save_pred=False)
    return p.parse_args()


def main():
    # args = _get_args()
    # X, y = npnn.load_mnist(args.dataset)

    BATCH_SIZE = 32
    N_EPOCHS = 25
    X, y = npnn.dataset.load_mnist('mnist.npz') # (dataset size, features) (dataset size, classes)
    X_train = X[:50000, :, :, :]
    y_train = y[:50000, :]
    X_valid = X[50000:, :, :, :]
    y_valid = y[50000:, :]
    print('Shapes: train:', X_train.shape, y_train.shape, 'valid:', X_valid.shape, y_valid.shape)
    train_dataset = npnn.dataset.Dataset(X_train, y_train, batch_size=BATCH_SIZE)
    valid_dataset = npnn.dataset.Dataset(X_valid, y_valid, batch_size=BATCH_SIZE)
    modules = [Flatten(),
               Dense(784, 256),
               Dense(256, 64),
               ELU(alpha=.9),
               Dense(64, 10)]

    model = Sequential(modules, SoftmaxCrossEntropy(), SGD(learning_rate=.1))

    total_train_loss = []
    total_train_accuracy = []
    total_valid_loss = []
    total_valid_accuracy = []

    for epoch in tqdm(range(N_EPOCHS), "Training..."):
        train_loss, train_accuracy = model.train(train_dataset)
        valid_loss, valid_accuracy = model.test(valid_dataset)
        total_train_loss.append(train_loss)
        total_train_accuracy.append(train_accuracy)
        total_valid_loss.append(valid_loss)
        total_valid_accuracy.append(valid_accuracy)

    # print("train", total_train_loss)
    # print("valid", total_valid_loss)
    plt.plot(range(N_EPOCHS), total_train_loss, label='Train Loss')
    plt.plot(range(N_EPOCHS), total_valid_loss, label='Validation Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and validation losses")
    plt.legend()
    plt.show()

    # print("train", total_train_accuracy)
    # print("valid", total_valid_accuracy)
    plt.plot(range(N_EPOCHS), total_train_accuracy, label='Train Accuracy')
    plt.plot(range(N_EPOCHS), total_valid_accuracy, label='Validation Accuracy')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training and validation accuracy")
    plt.legend()
    plt.show()

    # Save statistics to file.
    # We recommend that you save your results to a file, then plot them
    # separately, though you can also place your plotting code here.
    # if args.save_stats:
        # stats.to_csv("data/{}_{}.csv".format(args.opt, args.lr))

    # Save predictions.
    # if args.save_pred:
    X_test, _ = npnn.load_mnist("mnist_test.npz")
    y_pred = np.argmax(model.forward(X_test), axis=1).astype(np.uint8)
    np.save("mnist_test_pred.npy", y_pred)

    best_validations = []
    learning_rates = [.05, .1, .2, .5, 1.0]
    for rate in tqdm(learning_rates, desc = '=== OUTER ==='):
        valids = []
        model = Sequential(modules, SoftmaxCrossEntropy(), SGD(learning_rate=rate))
        for epoch in tqdm(range(N_EPOCHS), "Training..."):
            train_loss, train_accuracy = model.train(train_dataset)
            valid_loss, valid_accuracy = model.test(valid_dataset)
            valids.append(valid_loss)
            # total_train_loss.append(train_loss)
            # total_train_accuracy.append(train_accuracy)
            # total_valid_loss.append(valid_loss)
            # total_valid_accuracy.append(valid_accuracy)
        m = np.nanmin(valids)
        if m == float('nan'):
            raise RuntimeError(f"NaN value encountered! {rate}")
        best_validations.append(m)

    print(best_validations)
    plt.plot(learning_rates, best_validations, label="Validation Loss")
    plt.xlabel("Learning Rates")
    plt.ylabel("Validation Loss")
    plt.title("Validation loss across multiple different learning rates")
    plt.xscale('log')
    plt.xticks(learning_rates, learning_rates)
    plt.show()

if __name__ == '__main__':
    import traceback
    import importlib
    importlib.reload(npnn.modules)
    importlib.reload(npnn.model)
    try:
        main()
    except Exception as e:
        traceback.print_exception(e)
