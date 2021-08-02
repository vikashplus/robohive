import numpy as np

def calculate_cosine(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    """Calculates the cosine angle between two vectors.

    This computes cos(theta) = dot(v1, v2) / (norm(v1) * norm(v2))

    Args:
        vec1: The first vector. This can have a batch dimension.
        vec2: The second vector. This can have a batch dimension.

    Returns:
        The cosine angle between the two vectors, with the same batch dimension
        as the given vectors.
    """
    if np.shape(vec1) != np.shape(vec2):
        raise ValueError('{} must have the same shape as {}'.format(vec1, vec2))
    ndim = np.ndim(vec1)
    norm_product = (np.linalg.norm(vec1, axis=-1) * np.linalg.norm(vec2, axis=-1))
    zero_norms = norm_product == 0
    if np.any(zero_norms):
        if ndim>1:
            norm_product[zero_norms] = 1
        else:
            norm_product = 1
    # Return the batched dot product.
    return np.einsum('...i,...i', vec1, vec2) / norm_product