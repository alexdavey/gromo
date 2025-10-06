from warnings import warn

import torch


def sqrt_inverse_matrix_semi_positive(
    matrix: torch.Tensor,
    threshold: float = 1e-5,
    preferred_linalg_library: None | str = None,
) -> torch.Tensor:
    """
    Compute the square root of the inverse of a semi-positive definite matrix.

    Parameters
    ----------
    matrix: torch.Tensor
        input matrix, square and semi-positive definite
    threshold: float
        threshold to consider an eigenvalue as zero
    preferred_linalg_library: None | str in ("magma", "cusolver")
        linalg library to use, "cusolver" may fail
        for non-positive definite matrix if CUDA < 12.1 is used
        see: https://pytorch.org/docs/stable/generated/torch.linalg.eigh.html

    Returns
    -------
    torch.Tensor
        square root of the inverse of the input matrix
    """
    assert matrix.shape[0] == matrix.shape[1], "The input matrix must be square."
    assert torch.allclose(matrix, matrix.t()), "The input matrix must be symmetric."
    assert torch.isnan(matrix).sum() == 0, "The input matrix must not contain NaN values."

    if preferred_linalg_library is not None:
        torch.backends.cuda.preferred_linalg_library(preferred_linalg_library)
    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    except torch.linalg.LinAlgError as e:
        if preferred_linalg_library == "cusolver":
            raise ValueError(
                "This is probably a bug from CUDA < 12.1"
                "Try torch.backends.cuda.preferred_linalg_library('magma')"
            )
        else:
            raise e
    selected_eigenvalues = eigenvalues > threshold
    eigenvalues = torch.rsqrt(eigenvalues[selected_eigenvalues])  # inverse square root
    eigenvectors = eigenvectors[:, selected_eigenvalues]
    return eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.t()


def optimal_delta(
    tensor_s: torch.Tensor,
    tensor_m: torch.Tensor,
    dtype: torch.dtype = torch.float32,
    force_pseudo_inverse: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the optimal delta for the layer using current S and M tensors.

    dW* = S[-1]^-1 M (if needed we use the pseudo-inverse)

    Compute dW* (and dBias* if needed).
    L(A + gamma * B * dW) = L(A) - gamma * d + o(gamma)
    where d is the first order decrease and gamma the scaling factor.

    Parameters
    ----------
    tensor_s: torch.Tensor, of shape [total_in_features, total_in_features]
        S tensor from calling layer, shape
    tensor_m: torch.Tensor, of shape [total_in_features, in_features]
        M tensor from calling layer
    dtype: torch.dtype, default torch.float32
        dtype for S and M during the computation
    force_pseudo_inverse: bool, default False
        if True, use the pseudo-inverse to compute the optimal delta even if the
        matrix is invertible

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        the optimal delta weights and the first order decrease
    """
    # Ensure both tensors have the same dtype initially
    assert tensor_s.dtype == tensor_m.dtype, (
        f"Both input tensors must have the same dtype, "
        f"got tensor_s.dtype={tensor_s.dtype} and tensor_m.dtype={tensor_m.dtype}"
    )

    saved_dtype = tensor_s.dtype
    if tensor_s.dtype != dtype:
        tensor_s = tensor_s.to(dtype=dtype)
    if tensor_m.dtype != dtype:
        tensor_m = tensor_m.to(dtype=dtype)

    if not force_pseudo_inverse:
        try:
            delta_raw = torch.linalg.solve(tensor_s, tensor_m).t()
        except torch.linalg.LinAlgError:
            force_pseudo_inverse = True
            # self.delta_raw = torch.linalg.lstsq(tensor_s, tensor_m).solution.t()
            # do not use lstsq because it does not work with the GPU
            warn("Using the pseudo-inverse for the computation of the optimal delta.")
    if force_pseudo_inverse:
        delta_raw = (torch.linalg.pinv(tensor_s) @ tensor_m).t()

    assert delta_raw is not None, "delta_raw should be computed by now."
    assert (
        delta_raw.isnan().sum() == 0
    ), "The optimal delta should not contain NaN values."
    parameter_update_decrease = torch.trace(tensor_m @ delta_raw)
    if parameter_update_decrease < 0:
        warn(
            "The parameter update decrease should be positive, "
            f"but got {parameter_update_decrease=} for layer."
        )
        if not force_pseudo_inverse:
            warn("Trying to use the pseudo-inverse with torch.float64.")
            return optimal_delta(
                tensor_s, tensor_m, dtype=torch.float64, force_pseudo_inverse=True
            )
        else:
            warn("Failed to compute the optimal delta, set delta to zero.")
            delta_raw.fill_(0)
            parameter_update_decrease.fill_(0)
    delta_raw = delta_raw.to(dtype=saved_dtype)
    if isinstance(parameter_update_decrease, torch.Tensor):
        parameter_update_decrease = parameter_update_decrease.to(dtype=saved_dtype)

    return delta_raw, parameter_update_decrease


def compute_optimal_added_parameters(
    matrix_s: torch.Tensor,
    matrix_n: torch.Tensor,
    numerical_threshold: float = 1e-15,
    statistical_threshold: float = 1e-3,
    maximum_added_neurons: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the optimal added parameters for a given layer.

    Parameters
    ----------
    matrix_s: torch.Tensor in (s, s)
        square matrix S
    matrix_n: torch.Tensor in (s, t)
        matrix N
    numerical_threshold: float
        threshold to consider an eigenvalue as zero in the square root of the inverse of S
    statistical_threshold: float
        threshold to consider an eigenvalue as zero in the SVD of S{-1/2} N
    maximum_added_neurons: int | None
        maximum number of added neurons, if None all significant neurons are kept

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor] in (k, s) (t, k) (k,)
        optimal added weights alpha, omega and eigenvalues lambda
    """
    # matrix_n = matrix_n.t()
    s_1, s_2 = matrix_s.shape
    assert s_1 == s_2, "The input matrix S must be square."
    n_1, n_2 = matrix_n.shape
    assert s_2 == n_1, (
        f"The input matrices S and N must have compatible shapes."
        f"(got {matrix_s.shape=} and {matrix_n.shape=})"
    )
    if not torch.allclose(matrix_s, matrix_s.t()):
        diff = torch.abs(matrix_s - matrix_s.t())
        warn(
            f"Warning: The input matrix S is not symmetric.\n"
            f"Max difference: {diff.max():.2e},\n"
            f"% of non-zero elements: {100 * (diff > 1e-10).sum() / diff.numel():.2f}%"
        )
        matrix_s = (matrix_s + matrix_s.t()) / 2

    # assert torch.allclose(matrix_s, matrix_s.t()), "The input matrix S must be symmetric."

    # compute the square root of the inverse of S
    matrix_s_inverse_sqrt = sqrt_inverse_matrix_semi_positive(
        matrix_s, threshold=numerical_threshold
    )
    # compute the product P := S^{-1/2} N
    matrix_p = matrix_s_inverse_sqrt @ matrix_n
    # compute the SVD of the product
    try:
        u, s, v = torch.linalg.svd(matrix_p, full_matrices=False)
    except torch.linalg.LinAlgError:
        print("Warning: An error occurred during the SVD computation.")
        print(f"matrix_s: {matrix_s.min()=}, {matrix_s.max()=}, {matrix_s.shape=}")
        print(f"matrix_n: {matrix_n.min()=}, {matrix_n.max()=}, {matrix_n.shape=}")
        print(
            f"matrix_s_inverse_sqrt: {matrix_s_inverse_sqrt.min()=}, {matrix_s_inverse_sqrt.max()=}, {matrix_s_inverse_sqrt.shape=}"
        )
        print(f"matrix_p: {matrix_p.min()=}, {matrix_p.max()=}, {matrix_p.shape=}")
        u, s, v = torch.linalg.svd(matrix_p, full_matrices=False)
        # raise ValueError("An error occurred during the SVD computation.")

        # u = torch.zeros((1, matrix_p.shape[0]))
        # s = torch.zeros(1)
        # v = torch.randn((matrix_p.shape[1], 1))
        # return u, v, s

    # select the singular values
    selected_singular_values = s >= min(statistical_threshold, s.max())
    if maximum_added_neurons is not None:
        selected_singular_values[maximum_added_neurons:] = False

    # keep only the significant singular values but keep at least one
    s = s[selected_singular_values]
    u = u[:, selected_singular_values]
    v = v[selected_singular_values, :]
    # compute the optimal added weights
    sqrt_s = torch.sqrt(torch.abs(s))
    alpha = torch.sign(s) * sqrt_s * (matrix_s_inverse_sqrt @ u)
    omega = sqrt_s[:, None] * v
    return alpha.t(), omega.t(), s
