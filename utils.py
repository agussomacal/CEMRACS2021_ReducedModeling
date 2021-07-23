import itertools
from multiprocessing import Pool, cpu_count
from typing import List, Union, Callable

import numpy as np
import scipy.linalg as linalg
from scipy.interpolate import lagrange
import matplotlib.pylab as plt


# ----------------------------------------
# visualization utils
def squared_subplots(N_subplots, axes_xy_proportions=(4, 4)):
    if N_subplots > 0:
        nrows = int(np.sqrt(N_subplots))
        ncols = int(np.ceil(N_subplots / nrows))
        # ncols = int(np.sqrt(N_subplots))
        # nrows = int(np.ceil(N_subplots / ncols))
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True,
                               figsize=(axes_xy_proportions[0] * ncols, axes_xy_proportions[1] * nrows))
        if N_subplots == 1:
            ax = np.array(ax).reshape((1, 1))
        if len(ax.shape) == 1:
            ax = ax.reshape((1, -1))
        for i, j in itertools.product(np.arange(nrows), np.arange(ncols)):
            yield ax[i, j]


def plot_solution(ax, x, y, u_reshaped, contour_levels=0, vmin=None, vmax=None):
    if contour_levels:
        ax.contourf(x, y, u_reshaped, levels=contour_levels)
    else:
        h = ax.imshow(u_reshaped, vmin=vmin, vmax=vmax)
        plt.colorbar(h)
    ax.axvline(0.5, linestyle="dashed", alpha=0.7, color="gray")
    ax.axhline(0.5, linestyle="dashed", alpha=0.7, color="gray")


def plot_solutions_together(sm, diffusion_coefficients, solutions, num_points_per_dim_to_plot=100, contour_levels=0):
    x, y = np.meshgrid(*[np.linspace(0, 1, num=num_points_per_dim_to_plot)] * 2)
    for i, (ax, a, u) in enumerate(zip(squared_subplots(len(solutions)), diffusion_coefficients, solutions)):
        u = sm.evaluate_solutions(np.concatenate((x.reshape((-1, 1)), y.reshape((-1, 1))), axis=1), solutions=[u])
        ax.set_title("a={}".format(np.round(np.reshape(a, (2, 2)), decimals=2)))
        plot_solution(ax, x, y, u.reshape((num_points_per_dim_to_plot, num_points_per_dim_to_plot)), contour_levels)


def plot_approximate_solutions_together(sm, diffusion_coefficients, solutions, approximate_solutions,
                                        num_points_per_dim_to_plot=100, contour_levels=0, measurement_points=None):
    x, y = np.meshgrid(*[np.linspace(0, 1, num=num_points_per_dim_to_plot)] * 2)
    for i, (a, u_aprox, u_true) in enumerate(zip(diffusion_coefficients, approximate_solutions, solutions)):
        ua = sm.evaluate_solutions(np.concatenate((x.reshape((-1, 1)), y.reshape((-1, 1))), axis=1),
                                   solutions=[u_aprox])
        ut = sm.evaluate_solutions(np.concatenate((x.reshape((-1, 1)), y.reshape((-1, 1))), axis=1), solutions=[u_true])

        fig, ax = plt.subplots(ncols=2)
        fig.suptitle("State estimation of \n a={}".format(np.round(np.reshape(a, (2, 2)), decimals=2)))

        vmin = np.min((np.min(ua), np.min(ut)))
        vmax = np.max((np.max(ua), np.max(ut)))
        plot_solution(ax[0], x, y, ua.reshape((num_points_per_dim_to_plot, num_points_per_dim_to_plot)), contour_levels,
                      vmin=vmin, vmax=vmax)
        plot_solution(ax[1], x, y, ut.reshape((num_points_per_dim_to_plot, num_points_per_dim_to_plot)), contour_levels,
                      vmin=vmin, vmax=vmax)

        ax[0].set_title("\n Approximation")
        ax[1].set_title("\n Solution")

        if measurement_points is not None:
            ax[1].scatter(*measurement_points.T, marker="x", alpha=0.8, s=5, color="white")

        # ax[2].imshow(a)
        # for i in range(np.shape(a)[0]):
        #     for j in range(np.shape(a)[1]):
        #         ax[2].text(j, i, a[i, j], ha="center", va="center", color="w")
        # ax[2].set_title("Diffusion coefficients")

# def plot_error():
#     for i, (ax, a, u_aprox, u_true) in enumerate(
#             zip(squared_subplots(number_of_test_solutions_to_plot), diffusion_coefficients_online,
#                 approximate_solutions, solutions_online)):
#         if i >= number_of_test_solutions_to_plot:
#             break
#         x, y = np.meshgrid(*[np.linspace(0, 1, num=num_points_per_dim_to_plot)] * 2)
#         ua = sm.evaluate_solutions(np.concatenate((x.reshape((-1, 1)), y.reshape((-1, 1))), axis=1),
#                                    solutions=[u_aprox])
#         ut = sm.evaluate_solutions(np.concatenate((x.reshape((-1, 1)), y.reshape((-1, 1))), axis=1), solutions=[u_true])
#
#         ax.set_title("State estimation of \n a={}".format(np.round(a, decimals=2)))
#         h = ax.imshow(ua.reshape((num_points_per_dim_to_plot, num_points_per_dim_to_plot)) - ut.reshape(
#             (num_points_per_dim_to_plot, num_points_per_dim_to_plot)))
#         plt.colorbar(h, ax=ax)
#         ax.set_title("Approximation")


# -------- forward function ---------- #
def galerkin(a, B_total, A_preassembled):
    # create A matrix
    A_assembled = np.zeros(np.shape(A_preassembled)[1:3])
    for quarter in range(4):
        A_assembled += np.tensordot(a[quarter], A_preassembled[quarter], axes=((0, 1), (2, 3)))

    coefs = linalg.solve(A_assembled, B_total, assume_a='sym')

    return coefs


def rotation(deg, points):
    return (np.array([[np.cos(deg), -np.sin(deg)], [np.sin(deg), np.cos(deg)]]) @ points.T).T


def get_vspace_dim_from_poly_degree(lagrange_polynomials_degree):
    return (2 * lagrange_polynomials_degree - 1) ** 2


def get_poly_degree_from_vspace_dim(vspace_dim):
    return int((np.sqrt(vspace_dim) + 1) / 2)


def init_polynomial_variables(lagrange_polynomials_degree):
    lagrange_polynomials_degree = lagrange_polynomials_degree
    quarter_dim = lagrange_polynomials_degree ** 2
    dim_1d = 2 * lagrange_polynomials_degree - 1
    vspace_dim = dim_1d ** 2
    center = lagrange_polynomials_degree * dim_1d - lagrange_polynomials_degree

    points = (1 + np.sin(np.linspace(-np.pi / 2, np.pi / 2, lagrange_polynomials_degree + 1))) / 2
    base_lagrange = [lagrange(points, line) for line in np.eye(lagrange_polynomials_degree + 1)]

    # Create P conversion matrix
    P = np.zeros((4, quarter_dim, vspace_dim))
    for i in range(quarter_dim):
        P[0, i, center - (i % lagrange_polynomials_degree) - dim_1d * (i // lagrange_polynomials_degree)] = 1
        P[1, i, center + (i % lagrange_polynomials_degree) - dim_1d * (i // lagrange_polynomials_degree)] = 1
        P[2, i, center - (i % lagrange_polynomials_degree) + dim_1d * (i // lagrange_polynomials_degree)] = 1
        P[3, i, center + (i % lagrange_polynomials_degree) + dim_1d * (i // lagrange_polynomials_degree)] = 1

    return quarter_dim, dim_1d, vspace_dim, base_lagrange, P


class SolutionsManager:
    def __init__(self, lagrange_polynomials_degree):
        self.lagrange_polynomials_degree = lagrange_polynomials_degree
        self.quarter_dim, self.dim_1d, self.vspace_dim, self.base_lagrange, self.P = init_polynomial_variables(
            lagrange_polynomials_degree)

    def generate_solutions(self, a2try):
        if not hasattr(self, "B_total"):
            # do only once

            # create A pre assembled matrix without a coefficient
            A_quarter = np.zeros((self.quarter_dim, self.quarter_dim, 2, 2))
            for i in range(self.quarter_dim):
                for j in range(self.quarter_dim):
                    int_x_dx_phi_i_dx_phi_j = np.polyval(
                        np.polyint(np.polyder(self.base_lagrange[i // self.lagrange_polynomials_degree]) * np.polyder(
                            self.base_lagrange[j // self.lagrange_polynomials_degree])), 1)
                    int_y_phi_i_phi_j = np.polyval(np.polyint(
                        self.base_lagrange[i % self.lagrange_polynomials_degree] * self.base_lagrange[
                            j % self.lagrange_polynomials_degree]), 1)
                    A_quarter[i, j, 0, 0] = int_x_dx_phi_i_dx_phi_j * int_y_phi_i_phi_j

                    int_x_phi_i_phi_j = np.polyval(np.polyint(
                        self.base_lagrange[i // self.lagrange_polynomials_degree] * self.base_lagrange[
                            j // self.lagrange_polynomials_degree]),
                        1)
                    int_y_dy_phi_i_dy_phi_j = np.polyval(
                        np.polyint(np.polyder(self.base_lagrange[i % self.lagrange_polynomials_degree]) * np.polyder(
                            self.base_lagrange[j % self.lagrange_polynomials_degree])), 1)
                    A_quarter[i, j, 1, 1] = int_x_phi_i_phi_j * int_y_dy_phi_i_dy_phi_j

                    int_x_phi_i_dx_phi_j = np.polyval(
                        np.polyint(self.base_lagrange[i // self.lagrange_polynomials_degree] * np.polyder(
                            self.base_lagrange[j // self.lagrange_polynomials_degree])),
                        1)
                    int_y_dy_phi_i_phi_j = np.polyval(
                        np.polyint(
                            np.polyder(self.base_lagrange[i % self.lagrange_polynomials_degree]) * self.base_lagrange[
                                j % self.lagrange_polynomials_degree]),
                        1)
                    A_quarter[i, j, 0, 1] = int_x_phi_i_dx_phi_j * int_y_dy_phi_i_phi_j

                    int_x_dx_phi_i_phi_j = np.polyval(
                        np.polyint(
                            np.polyder(self.base_lagrange[i // self.lagrange_polynomials_degree]) * self.base_lagrange[
                                j // self.lagrange_polynomials_degree]),
                        1)
                    int_y_phi_i_dy_phi_j = np.polyval(
                        np.polyint(self.base_lagrange[i % self.lagrange_polynomials_degree] * np.polyder(
                            self.base_lagrange[j % self.lagrange_polynomials_degree])),
                        1)
                    A_quarter[i, j, 1, 0] = int_x_dx_phi_i_phi_j * int_y_phi_i_dy_phi_j

            self.A_preassembled = np.zeros((4, self.vspace_dim, self.vspace_dim, 2, 2))
            for quarter in range(4):
                self.A_preassembled[quarter] = np.tensordot(self.P[quarter],
                                                            np.tensordot(self.P[quarter], A_quarter, axes=((0,), (0,))),
                                                            axes=((0,), (1,)))

            # create B matrix
            B_quarter = np.zeros(self.quarter_dim)
            for i in range(self.quarter_dim):
                B_quarter[i] = np.polyval(np.polyint(self.base_lagrange[i // self.lagrange_polynomials_degree]),
                                          1) * np.polyval(
                    np.polyint(self.base_lagrange[i % self.lagrange_polynomials_degree]), 1)

            self.B_total = np.zeros(self.vspace_dim)
            for quarter in range(4):
                self.B_total += np.dot(self.P[quarter].T, B_quarter)

        if len(np.shape(a2try)) == 2:  # convert to a matricial coefficient
            a2try = np.array([[np.eye(2) * a_quarter for a_quarter in a[[2, 3, 0, 1]]] for a in a2try])

        return np.array([galerkin(a, self.B_total, self.A_preassembled) for a in a2try])

    def evaluate_solutions(self, points: np.ndarray, solutions: List[np.ndarray]) -> np.ndarray:
        """

        :param points: (m, 2) array of points coordinates between [0, 1] where we want to measure.
        :param solutions: List of n solutions (coefficients) to use to evaluate in the points.
        :return: (n, m) array with the evaluations of the n solutions in the m points.
        """
        M = len(points)
        square_ix = np.sign(np.array(points // 0.5, dtype=int))  # sign to force 0 or 1 when given a point in the border
        point_in_square = np.abs(2 * np.array(points) - 1)[:, [1, 0]]  # point in the subdomain
        square_ix[:, 1] *= 2
        square_ix = np.sum(square_ix, axis=1)  # which square subdomain

        x_eval_poly_lagrange = np.array(
            [np.polyval(polynom_unidim, point_in_square[:, 0]) for polynom_unidim in self.base_lagrange])
        y_eval_poly_lagrange = np.array(
            [np.polyval(polynom_unidim, point_in_square[:, 1]) for polynom_unidim in self.base_lagrange])

        eval_point_quarter = np.zeros((self.quarter_dim, M))
        for i in range(self.quarter_dim):
            eval_point_quarter[i, :] = x_eval_poly_lagrange[i // self.lagrange_polynomials_degree, :] * \
                                       y_eval_poly_lagrange[i % self.lagrange_polynomials_degree, :]

        eval_points = np.zeros((self.vspace_dim, M))
        for j in range(M):
            eval_points[:, j] = self.P[square_ix[j], :, :].T @ eval_point_quarter[:, j]

        return np.array([eval_points.T @ solution for solution in solutions])


def state_estimation_pipeline(solutions_offline: List[np.ndarray],
                              solutions_online: List[np.ndarray],
                              reduced_basis_generator: Callable,
                              measurements_sampling_method: Callable, number_of_measures: int,
                              list_number_of_reduced_base_elements: Union[List, np.ndarray], noise=0) -> List[
    np.ndarray]:
    approximate_solutions = []

    measurement_points = measurements_sampling_method(number_of_measures)
    sm = SolutionsManager(lagrange_polynomials_degree=get_poly_degree_from_vspace_dim(np.shape(solutions_online)[1]))
    measurements_online = sm.evaluate_solutions(measurement_points, solutions=solutions_online) + \
                          np.random.normal(scale=noise)
    for number_of_reduced_base_elements in list_number_of_reduced_base_elements:
        reduced_basis = reduced_basis_generator(solutions_offline, number_of_reduced_base_elements)

        sm = SolutionsManager(lagrange_polynomials_degree=get_poly_degree_from_vspace_dim(np.shape(reduced_basis)[1]))
        measurements_reduced_basis = sm.evaluate_solutions(measurement_points, solutions=reduced_basis)

        coefficients = np.linalg.lstsq(measurements_reduced_basis.T, measurements_online.T)[0]
        approximate_solutions.append(coefficients.T @ np.array(reduced_basis))

    return approximate_solutions


def calculate_approximation_statistics(points, approximate_solutions_list, solutions_online):
    smt = SolutionsManager(lagrange_polynomials_degree=get_poly_degree_from_vspace_dim(np.shape(solutions_online)[1]))

    error_mean = []
    error_median = []
    error_sd = []
    error_min = []
    error_max = []
    for app_sol in approximate_solutions_list:
        sma = SolutionsManager(lagrange_polynomials_degree=get_poly_degree_from_vspace_dim(np.shape(app_sol)[1]))
        error_mean.append(np.mean(np.sqrt(np.mean((sma.evaluate_solutions(points, solutions=app_sol) -
                                                   smt.evaluate_solutions(points, solutions=solutions_online)) ** 2,
                                                  axis=0))))
        error_median.append(np.median(np.sqrt(np.mean((sma.evaluate_solutions(points, solutions=app_sol) -
                                                       smt.evaluate_solutions(points, solutions=solutions_online)) ** 2,
                                                      axis=0))))
        error_sd.append(np.std(np.sqrt(np.mean((sma.evaluate_solutions(points, solutions=app_sol) -
                                                smt.evaluate_solutions(points, solutions=solutions_online)) ** 2,
                                               axis=0))))
        error_min.append(np.min(np.sqrt(np.mean((sma.evaluate_solutions(points, solutions=app_sol) -
                                                 smt.evaluate_solutions(points, solutions=solutions_online)) ** 2,
                                                axis=0))))
        error_max.append(np.max(np.sqrt(np.mean((sma.evaluate_solutions(points, solutions=app_sol) -
                                                 smt.evaluate_solutions(points, solutions=solutions_online)) ** 2,
                                                axis=0))))

    return error_mean, error_median, error_sd, error_min, error_max


if __name__ == "__main__":
    sm = SolutionsManager(lagrange_polynomials_degree=5)
    a2try = np.ones((4, 4)) * 100 - np.eye(4) * 99
    solutions = sm.generate_solutions(a2try)

    x, y = np.meshgrid(*[np.linspace(0, 1, num=5)] * 2)
    u = sm.evaluate_solutions(np.concatenate((x.reshape((-1, 1)), y.reshape((-1, 1))), axis=1), solutions=solutions)

