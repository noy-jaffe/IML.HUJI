from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    x = np.random.normal(10, 1, 1000)
    random_vec = UnivariateGaussian(False)
    random_vec.fit(x)
    print((random_vec.mu_, random_vec.var_))

    # Question 2 - Empirically showing sample mean is consistent
    estimated_mean = []
    mu = 10
    ms = np.linspace(10, 1000, 100).astype(np.int)
    for m in ms:
        part_sample_mean = np.mean(x[:m])
        distance = np.abs(part_sample_mean - mu)
        estimated_mean.append(distance)

    fig = go.Figure([go.Scatter(x=ms, y=estimated_mean, mode='markers+lines', name=r'$\widehat\mu$')],
                    layout=go.Layout(
                        title=r"$\text{(2) Distance Between The Estimated And True Value Of The Expectation As Function Of Number Of Samples}$",
                        xaxis_title="$\\text{m- number of samples}$",
                        yaxis_title="r$\\text {distance}$ ",
                        height=300))
    fig.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    pdf = random_vec.pdf(x)
    fig = go.Figure([go.Scatter(x=x, y=pdf, mode='markers')],
                    layout=go.Layout(
                        title=r"$\text{(3) PDF}$",
                        xaxis_title="$\\text{sample x}$",
                        yaxis_title="r$\\text {f(x)}$ ",
                        height=300))
    fig.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mean = np.array([0, 0, 4, 0])
    cov = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
    x = np.random.multivariate_normal(mean, cov, 1000)
    random = MultivariateGaussian()
    random.fit(x)
    print(random.mu_)
    print(random.cov_)

    # Question 5 - Likelihood evaluation


    # # Question 6 - Maximum likelihood
    # raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    # test_univariate_gaussian()
    test_multivariate_gaussian()
