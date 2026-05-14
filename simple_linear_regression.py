import numpy as np

# Todo
#   - check if all the entries are equal so we dont divide by zero
#   - implement mse
#   - implement rmse
#   - implement mae
#   - adjusted r-squared


class SimpleLinearRegression:
    """
    This class describes the main Simple Linear Regression Model
        - Calculates B0 and B1
        - Calculates the estimated y (yhat)
        - Calcululates RSS
        - Calculates R-Squared
    """

    def __init__(self):
        self.b0_hat = None
        self.b1_hat = None
        self.y_hat = None

    def b1(self, xi: np.array, yi: np.array) -> float:
        x_mean = np.mean(xi)
        return np.sum((xi - x_mean) * yi) / np.sum((xi - x_mean) * xi)

    def b0(self, b1: float, xi: np.array, yi: np.array) -> float:
        y_mean = np.mean(yi)
        return y_mean - b1 * np.mean(xi)

    def rss(self, yi: np.array, yhat: np.array) -> float:
        return np.sum((yi - yhat) ** 2)

    def rsquared(self, yi: np.array, yhat: np.array) -> float:
        y_mean = np.mean(yi)
        return 1 - (np.sum((yi - yhat) ** 2)) / np.sum((yi - y_mean) ** 2)

    def yhat(self, b0: float, b1: float, xi: np.array) -> np.array:
        return b0 + b1 * xi

    def standard_errors(self, b0: float, b1: float) -> float:
        pass

    def fit(self, X_train, y_train):
        self.b1_hat = self.b1(X_train, y_train)
        self.b0_hat = self.b0(self.b1_hat, X_train, y_train)
        self.y_hat = self.yhat(self.b0_hat, self.b1_hat, X_train)
        rsquared = self.rsquared(y_train, self.y_hat)
        print(f"The R2 of the fit {rsquared}")
        return self.y_hat

    def score(self, X_test: np.array, y_test: np.array) -> float:
        y_pred = self.predict(X_test)
        return self.rsquared(y_test, y_pred)

    def predict(self, X_test: np.array) -> np.array:
        y_pred = self.yhat(self.b0_hat, self.b1_hat, X_test)
        return y_pred


if __name__ == "__main__":
    X_train = np.array([1, 2, 4, 5, 6, 6, 7, 7])
    y_train = np.array([3, 7.6, 9.5, 9.8, 11, 11, 12.75, 12.75])
    X_test = np.array([3, 8, 9])
    y_test = np.array([6, 13.3, 11.5])

    lr = SimpleLinearRegression()

    lr.fit(X_train, y_train)

    rsquared = lr.score(X_test, y_test)

    # if r2 is negative it means it is worse that just predicting the average of yi

    print(f"The R2 of the fit: {rsquared}")
