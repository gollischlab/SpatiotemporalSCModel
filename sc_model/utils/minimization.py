import numpy as np
from scipy.optimize import (
    check_grad,
    minimize,
)


def fit_parameters_mle(
    input_signal,
    response_counts,
    fit_func,
    starter_params,
    method="L-BFGS-B",
    fit_func_der=None,
    bounds=None,
    check_gradient=False,
    options=None,
):
    """
    Fit parameters using maximum likelihood estimation (MLE) for Poisson spike trains.

    This function minimizes the negative log-likelihood function for Poisson spike trains
    using the provided fit function and its derivative. It uses the `scipy.optimize.minimize`
    function to perform the optimization.

    The function takes the input signal, response counts, fit function, initial parameters,
    and other optional parameters for the optimization. It returns the optimized parameters
    that minimize the negative log-likelihood function.

    :param input_signal:
        The input signal to the nonlinearity defined in the `fit-func` function.
    :param response_counts:
        The response counts that the output of the `fit-func` function should be fitted to.
    :param fit_func:
        The non-linear activation function to be used in the model.
    :param starter_params:
        The initial parameters for the optimization.
    :param method:
        The optimization method to be used. This argument is passed directly to
        `scipy.optimize.minimize`. Default is "L-BFGS-B".
    :param fit_func_der:
        The derivative of the non-linear activation function with respect to the parameters to be optimized.
        This function is used to compute the gradient of the negative log-likelihood function.
        If None, the gradient is not computed.
    :param bounds:
        The bounds for the parameters to be optimized.
        The bounds should be specified as a list of tuples, where each tuple contains the lower and upper
        bounds for each parameter. For example, [(0, 1), (0, None), (0, None)] specifies that the first
        parameter is bounded between 0 and 1, while the second and third parameters are unbounded.
    :param check_gradient:
        If True, the gradient of the negative log-likelihood function is checked using finite differences.
        This is useful for debugging and ensuring that the gradient is computed correctly.
        Default is False.
    :param options:
        Additional options for the optimization algorithm. This argument is passed directly to
        `scipy.optimize.minimize`. For example, you can specify the maximum number of iterations or
        the tolerance for convergence.

    :return:
        The optimized parameters that minimize the negative log-likelihood function.
        The output is a list of the fitted parameters.
    """
    loglike_fn = poisson_negative_log_likelihood

    if fit_func_der is not None:
        loglike_prime_fn = poisson_negative_log_likelihood_derivative
    else:
        loglike_prime_fn = None

    if check_gradient and loglike_prime_fn is not None:
        grads = []
        for i in range(10):
            grad_error = check_grad(
                loglike_fn,
                loglike_prime_fn,
                np.random.uniform(0, 1, len(starter_params)),
                fit_func,
                fit_func_der,
                input_signal,
                response_counts,
            )
            grads.append(grad_error)

        print(f"Gradient error: {np.mean(grads)} +- {np.std(grads)}")

    result = minimize(
        fun=loglike_fn,
        jac=loglike_prime_fn,
        x0=starter_params,
        args=(
            fit_func,
            fit_func_der,
            input_signal,
            response_counts,
        ),
        bounds=bounds,
        method=method,
        options=options
    )

    return result.x


def poisson_negative_log_likelihood(
    function_parameters: list,
    *args,
):
    """
    General negative log-likelihood function for poisson spike trains.

    :param function_parameters:
        The parameters of the non-linear activation function.
        These parameters are passed to the nonlinear function.
    :param args:
        Other arguments for the negative log-likelihood function.
        These include the nonlinear function, its derivative, the input signal, and the response counts.

    :return:
        The negative log-likelihood value for the given parameters.
        This value is computed under the assumption of a Poisson spiking process.
    """
    function, function_derivative, input_signal, response_counts = args

    # for reference in this function:
    #
    # M is the size of `response counts`
    # shape of input signal: (..., M)
    # there are no requirements on the first dimension of the `input signal`
    # and this must be taken care of by the `function`
    #
    # function output shape: (M,)
    function_output = function(input_signal, function_parameters)
    sum_1 = function_output.sum()
    sum_2 = (response_counts * np.log(function_output)).sum()
    return (sum_1 - sum_2) / response_counts.sum()


def poisson_negative_log_likelihood_derivative(
    function_parameters: list,
    *args,
):
    """
    First derivative of the negative log-likelihood function for poisson spike trains.

    :param function_parameters:
        The parameters of the non-linear activation function.
        These parameters are passed to the nonlinear function and its derivative.
    :param args:
        Other arguments for the negative log-likelihood function.
        These include the nonlinear function, its derivative, the input signal, and the response counts.
    :return:
    """
    function, function_derivative, input_signal, response_counts = args

    # for reference in this function:
    #
    # M is the size of `response counts`
    # N is the number of independent function parameters
    # shape of input signal: (..., M)
    # there are no requirements on the first dimension of the `input signal`
    # and this must be taken care of by the `function` and `function derivative`
    #
    # derivative output shape: (M, N)
    derivative_output = function_derivative(input_signal, function_parameters)
    # function output shape: (M,)
    function_output = function(input_signal, function_parameters)
    # sum_1 shape: (N,)
    sum_1 = derivative_output.sum(axis=0)
    # sum_2 shape: (N,)
    #                      (M,)                 @      (M, N)
    sum_2 = (response_counts / function_output) @ derivative_output
    return (sum_1 - sum_2) / response_counts.sum()
