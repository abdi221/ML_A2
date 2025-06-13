from MachineLearningModel import MachineLearningModel
import numpy as np
import matplotlib.pyplot as plt


class RegressionModelNormalEquation(MachineLearningModel):
    
    def __init__(self, degree=1):
        self.degree = degree
        self.theta: np.ndarray
        self.cost_history: list[float] = []
    
    def prepare_features(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        cols = [np.ones((X.shape[0], 1))]
        for k in range(1, self.degree + 1):
            cols.append(X**k)
        return np.concatenate(cols, axis=1)
        
    # https://numpy.org/doc/2.1/reference/generated/numpy.matmul.html
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        Xp = self.prepare_features(X)
        self.theta = np.linalg.inv(np.matmul(Xp.T, Xp)).dot(np.matmul(Xp.T, y))
        resid = np.dot(Xp, self.theta) - y
        mse = np.dot(resid, resid) / len(y)
        self.cost_history = [float(mse)]
        

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.theta is None:
            raise ValueError("Model not fitted yet...")
        Xp = self.prepare_features(X)
        return np.matmul(Xp, self.theta)

    def evaluate(self, X:np.ndarray, y:np.ndarray):
        y_hat = self.predict(X)
        return float(np.mean((y_hat -y)**2))
    
    def get_params(self) -> np.ndarray:
        return self.theta.copy()
    
    def get_cost_history(self) -> list[float]:
        return self.cost_history.copy()
    

if __name__ == "__main__":

    # load data
    df = np.genfromtxt("datasets/housing-boston.csv",
                       delimiter=",", autostrip=True, skip_header=1, dtype=float)
    # target_col = "MEDV" if "MEDV" in df.column else "PRICE"
    X = df[:, :2]
    y = df[:, 2]


    # train model
    model = RegressionModelNormalEquation(degree=1)
    model.fit(X, y)

    print("beta:", model.get_params())
    print("MSE:", model.evaluate(X, y))

    # scatter plots with regression lines
    fig, axes = plt.subplots(1, 2, figsize=(10,4))

    rm_values = X[:, 1]
    indus_values = X[:, 0]
    axes[0].scatter(rm_values, y, alpha=0.6, label="data")
    # regression line: vary RM, fix INDUS at its mean
    rm_grid = np.linspace(rm_values.min(), rm_values.max(), 100)
    indus_mean = indus_values.mean() * np.ones_like(rm_grid)
    y_hat_rm = model.predict(np.c_[indus_mean, rm_grid])
    axes[0].plot(rm_grid, y_hat_rm, label="regression", linewidth=2)
    axes[0].set_xlabel("RM")
    axes[0].set_ylabel("PRICE")
    axes[0].set_title("RM vs PRICE")
    axes[0].legend()

    # INDUS v Price/MEDV
    axes[1].scatter(rm_values, y, alpha=0.6, label="data")
    indus_grid = np.linspace(indus_values.min(), indus_values.max(), 100)
    rm_mean = rm_values.mean() * np.ones_like(indus_grid)
    y_hat_indus = model.predict(np.c_[indus_grid, rm_mean])
    axes[1].plot(indus_grid, model.predict(np.c_[indus_grid, rm_mean]), linewidth=2, label="regression")
    axes[1].set_xlabel("INDUS")
    axes[1].set_ylabel("PRICE")
    axes[1].set_title("INDUS vs Price")
    axes[1].legend()

    plt.tight_layout()
    plt.show()