from .fast_ops import FastOps
from .tensor_functions import rand, Function
from . import operators


def tile(input, kernel):
    """
    Reshape an image tensor for 2D pooling

    Args:
        input (:class:`Tensor`): batch x channel x height x width
        kernel ( pair of ints ): height x width of pooling

    Returns:
        (:class:`Tensor`, int, int) : Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.
    """

    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # TODO: Implement for Task 4.3.
    new_height = int(width / kw)
    new_width = int(height / kh)
    tnsr = input.contiguous()
    tnsr = tnsr.view(batch, channel, height, new_height, kw)
    tnsr_p = tnsr.permute(0, 1, 3, 2, 4)
    tnsr_p = tnsr_p.contiguous()
    tnsr_p = tnsr_p.view(batch, channel, new_height, new_width, (kw * kh))
    return tnsr_p, new_height, new_width


def avgpool2d(input, kernel):
    """
    Tiled average pooling 2D

    Args:
        input (:class:`Tensor`): batch x channel x height x width
        kernel ( pair of ints ): height x width of pooling

    Returns:
        :class:`Tensor` : pooled tensor
    """
    batch, channel, height, width = input.shape
    # TODO: Implement for Task 4.3.
    tnsr, new_height, new_width = tile(input, kernel)
    tnsr = tnsr.mean(4).view(batch, channel, new_height, new_width)
    return tnsr


max_reduce = FastOps.reduce(operators.max, -1e9)


def argmax(input, dim):
    """
    Compute the argmax as a 1-hot tensor.

    Args:
        input (:class:`Tensor`): input tensor
        dim (int): dimension to apply argmax


    Returns:
        :class:`Tensor` : tensor with 1 on highest cell in dim, 0 otherwise

    """
    out = max_reduce(input, [dim])
    return out == input


class Max(Function):
    @staticmethod
    def forward(ctx, input, dim):
        "Forward of max should be max reduction"
        # TODO: Implement for Task 4.4.
        ctx.save_for_backward(input, dim)
        return max_reduce(input, [dim])

    @staticmethod
    def backward(ctx, grad_output):
        "Backward of max should be argmax (see above)"
        # TODO: Implement for Task 4.4.
        input, dim = ctx.saved_values
        return argmax(input, dim) * grad_output


max = Max.apply


def softmax(input, dim):
    r"""
    Compute the softmax as a tensor.

    .. math::

        z_i = \frac{e^{x_i}}{\sum_i e^{x_i}}

    Args:
        input (:class:`Tensor`): input tensor
        dim (int): dimension to apply softmax

    Returns:
        :class:`Tensor` : softmax tensor
    """
    # TODO: Implement for Task 4.4.
    return input.exp() / input.exp().sum(dim)


def logsoftmax(input, dim):
    r"""
    Compute the log of the softmax as a tensor.

    .. math::

        z_i = x_i - \log \sum_i e^{x_i}

    See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations

    Args:
        input (:class:`Tensor`): input tensor
        dim (int): dimension to apply log-softmax

    Returns:
        :class:`Tensor` : log of softmax tensor
    """
    # TODO: Implement for Task 4.4.
    x = input.exp().sum(dim).log()
    return input - x


def maxpool2d(input, kernel):
    """
    Tiled max pooling 2D

    Args:
        input (:class:`Tensor`): batch x channel x height x width
        kernel ( pair of ints ): height x width of pooling

    Returns:
        :class:`Tensor` : pooled tensor
    """
    batch, channel, height, width = input.shape
    # TODO: Implement for Task 4.4.
    tnsr, new_height, new_width = tile(input, kernel)
    tnsr = max(tnsr, 4).contiguous().view(batch, channel, new_height, new_width)
    return tnsr


def dropout(input, rate, ignore=False):
    """
    Dropout positions based on random noise.

    Args:
        input (:class:`Tensor`): input tensor
        rate (float): probability [0, 1) of dropping out each position
        ignore (bool): skip dropout, i.e. do nothing at all

    Returns:
        :class:`Tensor` : tensor with random positions dropped out
    """
    # TODO: Implement for Task 4.4.
    if not ignore:
        selected = rand(input.shape)
        drop = selected > rate
        return input * drop
    else:
        return input
