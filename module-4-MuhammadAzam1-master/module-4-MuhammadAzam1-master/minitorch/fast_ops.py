import numpy as np
from .tensor_data import (
    count,
    index_to_position,
    broadcast_index,
    shape_broadcast,
    MAX_DIMS,
)
from numba import njit, prange


# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
count = njit(inline="always")(count)
index_to_position = njit(inline="always")(index_to_position)
broadcast_index = njit(inline="always")(broadcast_index)


def tensor_map(fn):
    """
    NUMBA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

    Args:
        fn: function mappings floats-to-floats to apply.
        out (array): storage for out tensor.
        out_shape (array): shape for out tensor.
        out_strides (array): strides for out tensor.
        in_storage (array): storage for in tensor.
        in_shape (array): shape for in tensor.
        in_strides (array): strides for in tensor.

    Returns:
        None : Fills in `out`
    """

    def _map(out, out_shape, out_strides, in_storage, in_shape, in_strides):
        # TODO: Implement for Task 3.1.
        if (len(out_strides) != len(in_strides) or (out_shape != in_shape).any() or (out_strides != in_strides).any()):
            for i in prange(len(out)):
                y = np.empty(MAX_DIMS, np.int32)
                x = np.empty(MAX_DIMS, np.int32)
                count(i, out_shape, y)
                broadcast_index(y, out_shape, in_shape, x)
                x_pos = index_to_position(x, in_strides)
                y_pos = index_to_position(y, out_strides)
                data = in_storage[x_pos]
                out[y_pos] = fn(data)
        else:
            for i in prange(len(out)):
                data = in_storage[i]
                out[i] = fn(data)

    return njit(parallel=True)(_map)


def map(fn):
    """
    Higher-order tensor map function ::

      fn_map = map(fn)
      b = fn_map(a)


    Args:
        fn: function from float-to-float to apply.
        a (:class:`Tensor`): tensor to map over
        out (:class:`Tensor`): optional, tensor data to fill in,
               should broadcast with `a`

    Returns:
        :class:`Tensor` : new tensor
    """

    # This line JIT compiles your tensor_map
    f = tensor_map(njit()(fn))

    def ret(a, out=None):
        if out is None:
            out = a.zeros(a.shape)
        f(*out.tuple(), *a.tuple())
        return out

    return ret


def tensor_zip(fn):
    """
    NUMBA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
        fn: function maps two floats to float to apply.
        out (array): storage for `out` tensor.
        out_shape (array): shape for `out` tensor.
        out_strides (array): strides for `out` tensor.
        a_storage (array): storage for `a` tensor.
        a_shape (array): shape for `a` tensor.
        a_strides (array): strides for `a` tensor.
        b_storage (array): storage for `b` tensor.
        b_shape (array): shape for `b` tensor.
        b_strides (array): strides for `b` tensor.

    Returns:
        None : Fills in `out`
    """

    def _zip(
        out,
        out_shape,
        out_strides,
        a_storage,
        a_shape,
        a_strides,
        b_storage,
        b_shape,
        b_strides,
    ):
        # TODO: Implement for Task 3.1.
        if (len(out_strides) != len(a_strides) or (out_shape != a_shape).any() or (out_strides != a_strides).any() or len(out_strides) != len(b_strides) or (out_shape != b_shape).any() or (out_strides != b_strides).any()):
            for i in prange(len(out)):
                y = np.empty(MAX_DIMS, np.int32)
                a = np.empty(MAX_DIMS, np.int32)
                b = np.empty(MAX_DIMS, np.int32)
                count(i, out_shape, y)
                broadcast_index(y, out_shape, a_shape, a)
                broadcast_index(y, out_shape, b_shape, b)
                y_pos = index_to_position(y, out_strides)
                a_pos = index_to_position(a, a_strides)
                b_pos = index_to_position(b, b_strides)
                a_s, b_s = a_storage[a_pos], b_storage[b_pos]
                out[y_pos] = fn(a_s, b_s)
        else:
            for i in prange(len(out)):
                a_s, b_s = a_storage[i], b_storage[i]
                out[i] = fn(a_s, b_s)

    return njit(parallel=True)(_zip)


def zip(fn):
    """
    Higher-order tensor zip function ::

      fn_zip = zip(fn)
      c = fn_zip(a, b)

    Args:
        fn: function from two floats-to-float to apply
        a (:class:`Tensor`): tensor to zip over
        b (:class:`Tensor`): tensor to zip over

    Returns:
        :class:`Tensor` : new tensor
    """
    f = tensor_zip(njit()(fn))

    def ret(a, b):
        c_shape = shape_broadcast(a.shape, b.shape)
        out = a.zeros(c_shape)
        f(*out.tuple(), *a.tuple(), *b.tuple())
        return out

    return ret


def tensor_reduce(fn):
    """
    NUMBA higher-order tensor reduce function.

    Args:
        fn: reduction function mapping two floats to float.
        out (array): storage for `out` tensor.
        out_shape (array): shape for `out` tensor.
        out_strides (array): strides for `out` tensor.
        a_storage (array): storage for `a` tensor.
        a_shape (array): shape for `a` tensor.
        a_strides (array): strides for `a` tensor.
        reduce_shape (array): shape of reduction (1 for dimension kept, shape value for dimensions summed out)
        reduce_size (int): size of reduce shape

    Returns:
        None : Fills in `out`

    """

    def _reduce(
        out,
        out_shape,
        out_strides,
        a_storage,
        a_shape,
        a_strides,
        reduce_shape,
        reduce_size,
    ):
        # TODO: Implement for Task 3.1.
        for i in prange(len(out)):
            y = np.empty(MAX_DIMS, np.int32)
            a = np.empty(MAX_DIMS, np.int32)
            count(i, out_shape, y)
            y_pos = index_to_position(y, out_strides)
            for j in range(reduce_size):
                count(j, reduce_shape, a)
                for k in range(len(reduce_shape)):
                    if reduce_shape[k] != 1:
                        y[k] = a[k]
                a_pos = index_to_position(y, a_strides)
                data = a_storage[a_pos]
                out[y_pos] = fn(out[y_pos], data)

    return njit(parallel=True)(_reduce)


def reduce(fn, start=0.0):
    """
    Higher-order tensor reduce function. ::

      fn_reduce = reduce(fn)
      reduced = fn_reduce(a, dims)


    Args:
        fn: function from two floats-to-float to apply
        a (:class:`Tensor`): tensor to reduce over
        dims (list, optional): list of dims to reduce
        out (:class:`Tensor`, optional): tensor to reduce into

    Returns:
        :class:`Tensor` : new tensor
    """

    f = tensor_reduce(njit()(fn))

    def ret(a, dims=None, out=None):
        old_shape = None
        if out is None:
            out_shape = list(a.shape)
            for d in dims:
                out_shape[d] = 1
            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start
        else:
            old_shape = out.shape
            diff = len(a.shape) - len(out.shape)
            out = out.view(*([1] * diff + list(old_shape)))

        # Assume they are the same dim
        assert len(out.shape) == len(a.shape)

        # Create a reduce shape / reduce size
        reduce_shape = []
        reduce_size = 1
        for i, s in enumerate(a.shape):
            if out.shape[i] == 1:
                reduce_shape.append(s)
                reduce_size *= s
            else:
                reduce_shape.append(1)

        # Apply
        f(*out.tuple(), *a.tuple(), np.array(reduce_shape), reduce_size)

        if old_shape is not None:
            out = out.view(*old_shape)
        return out

    return ret


@njit(parallel=True)
def tensor_matrix_multiply(
    out,
    out_shape,
    out_strides,
    a_storage,
    a_shape,
    a_strides,
    b_storage,
    b_shape,
    b_strides,
):
    """
    NUMBA tensor matrix multiply function.

    Should work for any tensor shapes that broadcast as long as ::

        assert a_shape[-1] == b_shape[-2]

    Args:
        out (array): storage for `out` tensor
        out_shape (array): shape for `out` tensor
        out_strides (array): strides for `out` tensor
        a_storage (array): storage for `a` tensor
        a_shape (array): shape for `a` tensor
        a_strides (array): strides for `a` tensor
        b_storage (array): storage for `b` tensor
        b_shape (array): shape for `b` tensor
        b_strides (array): strides for `b` tensor

    Returns:
        None : Fills in `out`
    """

    # TODO: Implement for Task 3.2.
    inside_indx = a_shape[-1]
    for i in prange(len(out)):
        y = np.empty(MAX_DIMS, np.int32)
        a = np.empty(MAX_DIMS, np.int32)
        b = np.empty(MAX_DIMS, np.int32)
        a_temp = np.empty(MAX_DIMS, np.int32)
        b_temp = np.empty(MAX_DIMS, np.int32)

        count(i, out_shape, y)
        count(i, out_shape, a_temp)

        a_temp[len(out_shape) - 1] = 0
        broadcast_index(a_temp, out_shape, a_shape, a)
        a_temp_idx = index_to_position(a, a_strides)

        count(i, out_shape, b_temp)
        b_temp[len(out_shape) - 2] = 0
        broadcast_index(b_temp, out_shape, b_shape, b)
        b_temp_idx = index_to_position(b, b_strides)

        total = 0.0
        out_position = index_to_position(y, out_strides)
        for k in range(inside_indx):
            total += (
                a_storage[a_temp_idx + k * a_strides[-1]]
                * b_storage[b_temp_idx + k * b_strides[-2]]
            )
        out[out_position] = total


def matrix_multiply(a, b):
    """
    Tensor matrix multiply

    Should work for any tensor shapes that broadcast in the first n-2 dims and
    have ::

        assert a.shape[-1] == b.shape[-2]

    Args:
        a (:class:`Tensor`): tensor a
        b (:class:`Tensor`): tensor b

    Returns:
        :class:`Tensor` : new tensor
    """

    # Create out shape
    # START CODE CHANGE
    ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
    ls.append(a.shape[-2])
    ls.append(b.shape[-1])
    assert a.shape[-1] == b.shape[-2]
    # END CODE CHANGE
    out = a.zeros(tuple(ls))

    # Call main function
    tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())
    return out


class FastOps:
    map = map
    zip = zip
    reduce = reduce
    matrix_multiply = matrix_multiply
