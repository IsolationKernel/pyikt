"""
pyikt (c) by Xin Han

pyikt is licensed under a
Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <https://creativecommons.org/licenses/by-nc-nd/4.0/>.
"""

import numpy as np
import scipy
from scipy import sparse


def delete_row_csr(mat, i):
    if not isinstance(mat, scipy.sparse.csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")
    n = mat.indptr[i + 1] - mat.indptr[i]
    if n > 0:
        mat.data[mat.indptr[i] : -n] = mat.data[mat.indptr[i + 1] :]
        mat.data = mat.data[:-n]
        mat.indices[mat.indptr[i] : -n] = mat.indices[mat.indptr[i + 1] :]
        mat.indices = mat.indices[:-n]
    mat.indptr[i:-1] = mat.indptr[i + 1 :]
    mat.indptr[i:] -= n
    mat.indptr = mat.indptr[:-1]
    mat._shape = (mat._shape[0] - 1, mat._shape[1])
    return mat


def safe_sparse_delete_row(mat, i):
    if sparse.issparse(mat):
        if mat.format != "csr":
            Warning("works only for CSR format -- use .tocsr()")
            mat = mat.tocsr()
        if isinstance(i, (list, np.ndarray)):
            for j in i:
                mat = delete_row_csr(mat, j)
            return mat
        return delete_row_csr(mat, i)
    else:
        return np.delete(mat, i, axis=0)
