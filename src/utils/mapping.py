from scipy.stats import linregress as lr
import scipy.optimize as so


def estimate_global_scaling(x, y):
    """
    Do least sq. curve fitting to estimate global scaling between
    two measurements.

    y' = scale * x

    :param x: Independent variable.
    :param y: Dependent variable.
    :return: The estimated global scaling.
    """
    out, _ = so.curve_fit(lambda i, m: m*i, x, y)
    return out[0]  # estimated scale


def estimate_global_scaling_intercept(x, y):
    """
    Perform linear regression between the two measurements.
    i.e. y' = scale * x + intercept

    :param x: Independent variable.
    :param y: Dependent variable.
    :return: scale and intercept of the regression line.
    """
    scale, intercept, _, _, _ = lr(x, y)
    return scale, intercept


def estimate_global_nonlinear(x, y):

    """
    :param x: Independent variable.
    :param y: Dependent variable.
    """


    pass

def estimate_global_nonlinear_with_bias(x, y):
    pass
