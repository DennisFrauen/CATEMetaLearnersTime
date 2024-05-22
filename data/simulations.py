from abc import ABC, abstractmethod
import numpy as np

class Simulator(ABC):
    def __init__(self, config):
        self.config = config

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    @abstractmethod
    def f_x(self, x_past, a_past, y_past):
        pass

    @abstractmethod
    def f_a(self, x_past, a_past, y_past):
        pass

    @abstractmethod
    def f_y(self, x_past, a_past, y_past):
        pass

    def simulate_factual(self, n):
        #initialize
        X = np.random.normal(loc=0, scale=1, size=(n, self.config["p"], self.config["T"]))
        A = np.zeros((n, 1, self.config["T"]), dtype=int)
        Y = np.zeros((n, 1, self.config["T"]))
        A[:, :, 0] = self.f_a(X[:, :, 0:1], None, None)
        Y[:, :, 0] = self.f_y(X[:, :, 0:1], A[:, :, 0:1], None)
        for t in range(1, self.config["T"]):
            X[:, :, t] = self.f_x(X[:, :, 0:t], A[:, :, 0:t], Y[:, :, 0:t])
            A[:, :, t] = self.f_a(X[:, :, 0:t+1], A[:, :, 0:t], Y[:, :, 0:t])
            Y[:, :, t] = self.f_y(X[:, :, 0:t+1], A[:, :, 0:t+1], Y[:, :, 0:t])

        active_entries = np.ones((n, self.config["T"], 1))
        X = np.transpose(X, (0, 2, 1))
        A = np.transpose(A, (0, 2, 1))
        Y = np.transpose(Y, (0, 2, 1))
        return X, A, Y, active_entries

    def simulate_intervention(self, n, a_int: list):
        # intervention horizon
        tau = len(a_int) - 1
        # intervention start time
        t_int = self.config["T"] - tau - 1
        # initialize
        X = np.random.normal(loc=0, scale=1, size=(n, self.config["p"], self.config["T"]))
        A = np.zeros((n, 1, self.config["T"]), dtype=int)
        Y = np.zeros((n, 1, self.config["T"]))
        if t_int > 0:
            A[:, :, 0] = self.f_a(X[:, :, 0:1], None, None)
        else:
            A[:, :, 0] = a_int[0]
        Y[:, :, 0] = self.f_y(X[:, :, 0:1], A[:, :, 0:1], None)


        for t in range(1, self.config["T"]):
            if t < t_int:
                X[:, :, t] = self.f_x(X[:, :, 0:t], A[:, :, 0:t], Y[:, :, 0:t])
                A[:, :, t] = self.f_a(X[:, :, 0:t + 1], A[:, :, 0:t], Y[:, :, 0:t])
                Y[:, :, t] = self.f_y(X[:, :, 0:t + 1], A[:, :, 0:t + 1], Y[:, :, 0:t])
            else:
                X[:, :, t] = self.f_x(X[:, :, 0:t], A[:, :, 0:t], Y[:, :, 0:t])
                A[:, :, t] = a_int[t-t_int]
                Y[:, :, t] = self.f_y(X[:, :, 0:t + 1], A[:, :, 0:t + 1], Y[:, :, 0:t])
        X = np.transpose(X, (0, 2, 1))
        A = np.transpose(A, (0, 2, 1))
        Y = np.transpose(Y, (0, 2, 1))
        active_entries = np.ones((n, self.config["T"], 1))
        return X, A, Y, active_entries



class Sim_Autoregressive(Simulator):
    def __init__(self, config):
        super().__init__(config)

    def adjust_weights(self, len_past, weights):
        if len(weights) > len_past:
            weights = weights[len(weights) - len_past:]
        elif len(weights) < len_past:
            weights = np.concatenate([np.zeros(len_past - len(weights)), weights])
        return weights

    # Functional assignments for sampling

    def f_x(self, x_past, a_past, y_past):
        weights_xx = self.adjust_weights(x_past.shape[2], self.config["weights_xx"])
        weights_ax = self.adjust_weights(x_past.shape[2], self.config["weights_ax"])
        weights_yx = self.adjust_weights(x_past.shape[2], self.config["weights_yx"])
        x_score = np.dot(x_past, weights_xx)
        a_score = np.dot(a_past - 0.5, weights_ax)
        y_score = np.dot(y_past, weights_yx)
        X_t = x_score + a_score + y_score + np.random.normal(loc=0, scale=self.config["noise_x"], size=(x_past.shape[0], x_past.shape[1]))
        return X_t

    def propensity(self, x_past, a_past, y_past):
        weights_xa = self.adjust_weights(x_past.shape[2], self.config["weights_xa"])
        weights_aa = self.adjust_weights(x_past.shape[2] - 1, self.config["weights_aa"])
        weights_ya = self.adjust_weights(x_past.shape[2] - 1, self.config["weights_ya"])
        x_score = np.mean(np.dot(x_past, weights_xa), axis=1, keepdims=True)
        if a_past is not None:
            a_score = np.dot(a_past - 0.5, weights_aa)
        else:
            a_score = 0
        if y_past is not None:
            y_score = np.dot(y_past, weights_ya)
        else:
            y_score = 0
        A_t_score = self.sigmoid(x_score - a_score + y_score)
        return A_t_score

    def f_a(self, x_past, a_past, y_past):
        A_t_score = self.propensity(x_past, a_past, y_past)
        A_t = np.random.binomial(1, A_t_score)
        return A_t

    def y_mean(self, x_past, a_past, y_past):
        weights_xy = self.adjust_weights(x_past.shape[2], self.config["weights_xy"])
        weights_ay = self.adjust_weights(x_past.shape[2], self.config["weights_ay"])
        weights_yy = self.adjust_weights(x_past.shape[2] - 1, self.config["weights_yy"])
        x_score = np.mean(np.dot(x_past, weights_xy), axis=1, keepdims=True)
        a_score = np.dot(a_past - 0.5, weights_ay)
        if y_past is not None:
            y_score = np.dot(y_past, weights_yy)
        else:
            y_score = 0
        score = x_score + a_score + y_score
       #effect_term = score * a_past[:, :, -1]
        response_term = x_score + y_score
        return np.cos(5 * response_term) + a_score

    def f_y(self, x_past, a_past, y_past):
        Y_t = self.y_mean(x_past, a_past, y_past) + np.random.normal(loc=0, scale=self.config["noise_y"], size=(x_past.shape[0], 1))
        return Y_t





class Sim_Autoregressive_Propensity(Simulator):
    def __init__(self, config):
        super().__init__(config)

    def adjust_weights(self, len_past, weights):
        if len(weights) > len_past:
            weights = weights[len(weights) - len_past:]
        elif len(weights) < len_past:
            weights = np.concatenate([np.zeros(len_past - len(weights)), weights])
        return weights

    # Functional assignments for sampling

    def f_x(self, x_past, a_past, y_past):
        weights_xx = self.adjust_weights(x_past.shape[2], self.config["weights_xx"])
        weights_ax = self.adjust_weights(x_past.shape[2], self.config["weights_ax"])
        weights_yx = self.adjust_weights(x_past.shape[2], self.config["weights_yx"])
        x_score = np.dot(x_past, weights_xx)
        a_score = np.dot(a_past - 0.5, weights_ax)
        y_score = np.dot(y_past, weights_yx)
        X_t = x_score + a_score + y_score + np.random.normal(loc=0, scale=self.config["noise_x"], size=(x_past.shape[0], x_past.shape[1]))
        return X_t

    def propensity(self, x_past, a_past, y_past):
        weights_xa = self.adjust_weights(x_past.shape[2], self.config["weights_xa"])
        weights_aa = self.adjust_weights(x_past.shape[2] - 1, self.config["weights_aa"])
        weights_ya = self.adjust_weights(x_past.shape[2] - 1, self.config["weights_ya"])
        x_score = np.mean(np.dot(x_past, weights_xa), axis=1, keepdims=True)
        if a_past is not None:
            a_score = np.dot(a_past - 0.5, weights_aa)
        else:
            a_score = 0
        if y_past is not None:
            y_score = np.dot(y_past, weights_ya)
        else:
            y_score = 0
        A_t_score = self.sigmoid(4*np.cos((-a_score + x_score + y_score)))
        return A_t_score

    def f_a(self, x_past, a_past, y_past):
        A_t_score = self.propensity(x_past, a_past, y_past)
        A_t = np.random.binomial(1, A_t_score)
        return A_t

    def y_mean(self, x_past, a_past, y_past):
        weights_xy = self.adjust_weights(x_past.shape[2], self.config["weights_xy"])
        weights_ay = self.adjust_weights(x_past.shape[2], self.config["weights_ay"])
        weights_yy = self.adjust_weights(x_past.shape[2] - 1, self.config["weights_yy"])
        x_score = np.mean(np.dot(x_past, weights_xy), axis=1, keepdims=True)
        a_score = np.dot(a_past - 0.5, weights_ay)
        if y_past is not None:
            y_score = np.dot(y_past, weights_yy)
        else:
            y_score = 0
        score = x_score + a_score + y_score
       #effect_term = score * a_past[:, :, -1]
        response_term = x_score + y_score
        return np.cos(response_term) + a_score

    def f_y(self, x_past, a_past, y_past):
        Y_t = self.y_mean(x_past, a_past, y_past) + np.random.normal(loc=0, scale=self.config["noise_y"], size=(x_past.shape[0], 1))
        return Y_t




class Sim_Autoregressive_Overlap(Simulator):
    def __init__(self, config):
        super().__init__(config)

    def adjust_weights(self, len_past, weights):
        if len(weights) > len_past:
            weights = weights[len(weights) - len_past:]
        elif len(weights) < len_past:
            weights = np.concatenate([np.zeros(len_past - len(weights)), weights])
        return weights

    # Functional assignments for sampling

    def f_x(self, x_past, a_past, y_past):
        weights_xx = self.adjust_weights(x_past.shape[2], self.config["weights_xx"])
        weights_ax = self.adjust_weights(x_past.shape[2], self.config["weights_ax"])
        weights_yx = self.adjust_weights(x_past.shape[2], self.config["weights_yx"])
        x_score = np.dot(x_past, weights_xx)
        a_score = np.dot(a_past - 0.5, weights_ax)
        y_score = np.dot(y_past, weights_yx)
        X_t = x_score + a_score + y_score + np.random.normal(loc=0, scale=self.config["noise_x"], size=(x_past.shape[0], x_past.shape[1]))
        return X_t

    def propensity(self, x_past, a_past, y_past):
        weights_xa = self.adjust_weights(x_past.shape[2], self.config["weights_xa"])
        weights_aa = self.adjust_weights(x_past.shape[2] - 1, self.config["weights_aa"])
        weights_ya = self.adjust_weights(x_past.shape[2] - 1, self.config["weights_ya"])
        x_score = np.mean(np.dot(x_past, weights_xa), axis=1, keepdims=True)
        if a_past is not None:
            a_score = np.dot(a_past - 0.5, weights_aa)
        else:
            a_score = 0
        if y_past is not None:
            y_score = np.dot(y_past, weights_ya)
        else:
            y_score = 0
        A_t_score = self.sigmoid(self.config["overlap"]*(-a_score + x_score + y_score))
        return A_t_score

    def f_a(self, x_past, a_past, y_past):
        A_t_score = self.propensity(x_past, a_past, y_past)
        A_t = np.random.binomial(1, A_t_score)
        return A_t

    def y_mean(self, x_past, a_past, y_past):
        weights_xy = self.adjust_weights(x_past.shape[2], self.config["weights_xy"])
        weights_ay = self.adjust_weights(x_past.shape[2], self.config["weights_ay"])
        weights_yy = self.adjust_weights(x_past.shape[2] - 1, self.config["weights_yy"])
        x_score = np.mean(np.dot(x_past, weights_xy), axis=1, keepdims=True)
        a_score = np.dot(a_past - 0.5, weights_ay)
        if y_past is not None:
            y_score = np.dot(y_past, weights_yy)
        else:
            y_score = 0
        score = x_score + a_score + y_score
       #effect_term = score * a_past[:, :, -1]
        response_term = x_score + y_score
        return np.cos(response_term) + a_score

    def f_y(self, x_past, a_past, y_past):
        Y_t = self.y_mean(x_past, a_past, y_past) + np.random.normal(loc=0, scale=self.config["noise_y"], size=(x_past.shape[0], 1))
        return Y_t

