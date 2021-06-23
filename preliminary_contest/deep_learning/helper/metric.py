import numpy as np


def metric(outputs, labels):
    def logloss(y_true, y_pred, eps=1e-15):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        assert (len(y_true) and len(y_true) == len(y_pred))

        p = np.clip(y_pred, eps, 1 - eps)

        log_loss = np.sum(- y_true * np.log(p) - (1 - y_true) * np.log(1 - p))

        return log_loss / len(y_true)

    allloss = []

    outputs_len = len(outputs)

    for i in range(outputs_len):
        loss = logloss(outputs[i], labels[i])

        allloss.append(loss)

    mlogloss = np.sum(allloss) / outputs_len

    return 1 - mlogloss
