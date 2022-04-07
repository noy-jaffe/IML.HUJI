from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    raw_data = pd.read_csv(filename)

    columns_to_omit = ['id', 'date', 'lat', 'long']
    columns_positive = ["sqft_living", "sqft_lot", "yr_built", "sqft_living15", "sqft_lot15", "grade", "price"]
    columns_zero_one = ['waterfront', 'view', 'yr_renovated', 'zipcode']

    raw_data = raw_data.dropna().drop_duplicates()

    raw_data = raw_data.drop(columns=columns_to_omit)
    raw_data = raw_data["condition"].isin(range(1, 6))

    for feature in columns_positive:
        raw_data = raw_data[raw_data[feature] > 0]

    raw_data['yr_built'] = ((raw_data['yr_built'] / 10).astype(int)) - 190

    # makes 0-1 range
    raw_data = pd.get_dummies(raw_data, prefix=['waterfront_', 'view_', 'yr_renovated_', 'zipcode_'],
                              columns=columns_zero_one)

    raw_data.insert(0, "intercept", 1, True)  # add intercept

    raw_data = raw_data[(raw_data >= 0).all(1)]
    prices = raw_data["price"]
    raw_data = raw_data.drop("price", 1)

    return raw_data, prices


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    prefixs = ['waterfront_', 'view_', 'yr_renovated_', 'zipcode_']
    for prefix in prefixs:
        remove_col = [c for c in X.columns if c.lower()[:len(prefix)] != prefix]
        X = X[remove_col]

    X = X.drop("intercept", 1)

    y_std = np.std(y)
    for index_col in X:
        col_std = np.std(X[index_col])
        cov = np.cov(X[index_col], y)[0, 1]
        pc = cov / (col_std * y_std)
        fig = go.Figure([go.Scatter(x=X[index_col], y=y, mode='markers')],
                        layout=go.Layout(
                            title=f"The Pearson Correlation between {index_col} and response is {pc}",
                            xaxis_title=f"{index_col}$",
                            yaxis_title="r$\\text {prices}$ ",
                            height=300))
        fig.write_image(output_path + f"{index_col} Pearson Correlation ")


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X, y = load_data("C:/Users/noija/PycharmProjects/IML.HUJI/datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, y)

    # Question 3 - Split samples into training- and testing sets.
    train_x, train_y, test_x, test_y = split_train_test(X, y, .75)
    training_set = (train_x, train_y)
    testing_set = (test_x, test_x)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    # #   1) Sample p% of the overall training data
    # #   2) Fit linear model (including intercept) over sampled set
    # #   3) Test fitted model over test set
    # #   4) Store average and variance of loss over test set
    # # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    average_loss_p = []
    mean_p = []
    p_values = range(10, 101)
    for p in p_values:
        fraction = p / 100
        loss_per_p = np.ndarray(10)
        for i in range(0, 10):
            temp_x = train_x.sample(frac=fraction)
            new_train_y = temp_x["price"].to_numpy()
            new_train_x = temp_x.drop(columns=["price"]).to_numpy()
            train_linear = LinearRegression(True)
            train_linear.fit(new_train_x, new_train_y)
            loss_per_p[i] = (train_linear.loss(test_x.to_numpy(), test_y.to_numpy()))
        average_loss_p.append(np.mean(loss_per_p))
        mean_p.append(np.std(loss_per_p))
        mean_p = np.array(mean_p)
        std_p = np.array(average_loss_p)
        fig = go.Figure([go.Scatter(x=p_values, y=mean_p, mode="markers+lines", name="Mean Prediction",
                                    marker=dict(color="blue", opacity=.7)),
                         go.Scatter(x=p_values, y=mean_p - (2 * std_p), fill=None, mode="lines",
                                    line=dict(color="lightgrey"), showlegend=False),
                         go.Scatter(x=p_values, y=mean_p + (2 * std_p), fill='tonexty', mode="lines",
                                    line=dict(color="lightgrey"), showlegend=False), ],

                        layout=go.Layout(
                            title_text=rf"$\text{{The Mean Loss as a Function of {p}% - With Noise }}\mathcal{{N}}\left(0,2\right)$",
                            xaxis={"title": f"{p}%"},
                            yaxis={"title": r"Mean Loss"}))
        fig.show()
