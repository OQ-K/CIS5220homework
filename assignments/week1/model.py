import numpy as np


class LinearRegression:
    """
    A linear regression model that fits data to get a closed form solution.
    """

    w: np.ndarray
    b: float

    def __init__(self):
        self.w = 0
        self.b = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> tuple:
        '''Gives the closed form solution for the linear regression problem.'''
        X_ones = np.ones(X.shape[0]).reshape(X.shape[0], 1)
        X_new = np.concatenate((X, X_ones), axis=1)
        weights = np.linalg.inv(X_new.T @ X_new) @ (X_new.T @ y)
        self.w = weights[:8]
        self.b = weights[-1]
        return (self.w, self.b)

    def predict(self, X: np.ndarray) -> np.ndarray:
        '''Predicts labels with input X and fitted parameters w and b.'''
        return X @ self.w + self.b


class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """

    def fit(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.0000001, epochs: int = 30000
    ) -> tuple:
        '''Fits data with gradient descent with learning rate defualt to 1e-7 and epochs 3e4'''
        self.w = np.zeros(X.shape[1])
        self.b = 0
        for i in range(epochs):
            y_hat = self.predict(X)
            dw = X.T @ (y_hat - y) / X.shape[0]
            self.w = self.w - lr * dw
            db = np.mean(y_hat - y)
            self.b = self.b - lr * db
        return (self.w, self.b)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """
        return X @ self.w + self.b
