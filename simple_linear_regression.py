import numpy as np


class SimpleLinearRegression:
    def __init__(self):
        b0 = 0
        b1 = 0

    def b1(self, xi: np.array, yi: np.array) -> float:
        x_mean = np.mean(xi)
        return np.sum((xi - x_mean) * yi) / np.sum((xi - x_mean) * xi)

    def b0(self, b1: float, xi: np.array, yi: np.array) -> float:
        y_mean = np.mean(yi)
        return y_mean - b1 * np.mean(xi)

    def rss(self, yi: np.array, yihat: np.array) -> float:
        return np.sum((yi - yihat) ** 2)

    def yihat(self, b0: float, b1: float, xi: np.array) -> np.array:
        return b0 + b1 * xi

    def fit(self, X_train, y_train):
        b1 = self.b1(X_train, y_train)
        b0 = self.b0(b1, X_train, y_train)
        yihat = self.yihat(b0, b1, X_train)
        rss = self.rss(y_train, yihat)
        rsquared = self.rsquared(y_train, yihat)
        print(f"Preds : {yihat}")
        print(f"Residual Sum of Squares: {rss}")
        print(f"R squared: {rsquared}")

    def rsquared(self, y_train: np.array, yihat: np.array) -> float:
        y_mean = np.mean(y_train)
        return (np.sum(y_train - y_mean) - np.sum((y_train - yihat) ** 2)) / np.sum(
            (y_train - y_mean) ** 2
        )

    def predict(self, X_test: np.array) -> np.array:
        # preds = self.yihat(b0, self.b1, X_test)
        # return preds
        pass


if __name__ == "__main__":
    X_train = np.array([1, 2])
    y_train = np.array([3, 7.6])
    X_test = np.array([3, 8])
    y_test = np.array([6, 13.3])
    lr = SimpleLinearRegression()
    lr.fit(X_train, y_train)
    lr.predict(X_test)
