"""
pyikt (c) by Xin Han

pyikt is licensed under a
Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <https://creativecommons.org/licenses/by-nc-nd/4.0/>.
"""

from typing import Union

import numpy as np
import scipy.sparse as sp


def get_degrees(input_matrix: sp.csr_matrix, transpose: bool = False) -> np.ndarray:
    """Get the vector of degrees of a graph.

    If the graph is directed, returns the out-degrees (number of successors). Set ``transpose=True``
    to get the in-degrees (number of predecessors).

    For a biadjacency matrix, returns the degrees of rows. Set ``transpose=True`` to get the degrees of columns.

    Parameters
    ----------
    input_matrix : sparse.csr_matrix
        Adjacency or biadjacency matrix.
    transpose :
        If ``True``, transpose the input matrix.
    Returns
    -------
    degrees : np.ndarray
        Array of degrees.
    """
    if transpose:
        matrix = sp.csr_matrix(input_matrix.T)
    else:
        matrix = input_matrix
    degrees = matrix.indptr[1:] - matrix.indptr[:-1]
    return degrees


def get_neighbors(
    input_matrix: sp.csr_matrix, node: int, transpose: bool = False
) -> np.ndarray:
    """Get the neighbors of a node.

    If the graph is directed, returns the vector of successors. Set ``transpose=True``
    to get the predecessors.

    For a biadjacency matrix, returns the neighbors of a row node. Set ``transpose=True``
    to get the neighbors of a column node.

    """
    if transpose:
        matrix = sp.csr_matrix(input_matrix.T)
    else:
        matrix = input_matrix
    neighbors = matrix.indices[matrix.indptr[node] : matrix.indptr[node + 1]]
    return neighbors


def check_format(
    input_matrix: Union[
        sp.csr_matrix,
        sp.csc_matrix,
        sp.coo_matrix,
        sp.lil_matrix,
        np.ndarray,
        np.matrix,
    ],
    allow_empty: bool = False,
) -> sp.csr_matrix:
    """Check whether the matrix is a NumPy array or a Scipy sparse matrix and return
    the corresponding Scipy CSR matrix.
    """
    formats = {
        sp.csr_matrix,
        sp.csc_matrix,
        sp.coo_matrix,
        sp.lil_matrix,
        np.ndarray,
        np.matrix,
    }
    if type(input_matrix) not in formats:
        raise TypeError(
            "The input matrix must be in Scipy sparse format or Numpy ndarray format."
        )
    input_matrix = sp.csr_matrix(input_matrix)
    if not allow_empty and input_matrix.nnz == 0:
        raise ValueError("The input matrix is empty.")
    return input_matrix
