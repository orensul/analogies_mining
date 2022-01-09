"""
Sparsemax functions / losses.
Several classes here copied & pasted from the OpenNMT-py codebase:
https://github.com/OpenNMT/OpenNMT-py/blob/eaade1dc05963c1cc638c0cf6f3f714d6a555c0e/onmt/modules/sparse_activations.py
and
https://github.com/OpenNMT/OpenNMT-py/blob/eaade1dc05963c1cc638c0cf6f3f714d6a555c0e/onmt/modules/sparse_losses.py
"""

import torch
from torch.autograd import Function
import torch.nn as nn

def aeq(*args):
    """
    Assert all arguments have the same value
    """
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in arguments), \
        "Not all arguments have the same value: " + str(args)

def _make_ix_like(input, dim=0):
    d = input.size(dim)
    rho = torch.arange(1, d + 1, device=input.device, dtype=input.dtype)
    view = [1] * input.dim()
    view[0] = -1
    return rho.view(view).transpose(0, dim)


def _threshold_and_support(input, dim=0):
    """Sparsemax building block: compute the threshold
    Args:
        input: any dimension
        dim: dimension along which to apply the sparsemax
    Returns:
        the threshold value
    """

    input_srt, _ = torch.sort(input, descending=True, dim=dim)
    input_cumsum = input_srt.cumsum(dim) - 1
    rhos = _make_ix_like(input, dim)
    support = rhos * input_srt > input_cumsum

    support_size = support.sum(dim=dim).unsqueeze(dim)
    tau = input_cumsum.gather(dim, support_size - 1)
    tau /= support_size.to(input.dtype)
    return tau, support_size

class SparsemaxFunction(Function):

    @staticmethod
    def forward(ctx, input, dim=0):
        """sparsemax: normalizing sparse transform (a la softmax)
        Parameters:
            input (Tensor): any shape
            dim: dimension along which to apply sparsemax
        Returns:
            output (Tensor): same shape as input
        """
        ctx.dim = dim
        max_val, _ = input.max(dim=dim, keepdim=True)
        input -= max_val  # same numerical stability trick as for softmax
        tau, supp_size = _threshold_and_support(input, dim=dim)
        output = torch.clamp(input - tau, min=0)
        ctx.save_for_backward(supp_size, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        supp_size, output = ctx.saved_tensors
        dim = ctx.dim
        grad_input = grad_output.clone()
        grad_input[output == 0] = 0

        v_hat = grad_input.sum(dim=dim) / supp_size.to(output.dtype).squeeze()
        v_hat = v_hat.unsqueeze(dim)
        grad_input = torch.where(output != 0, grad_input - v_hat, grad_input)
        return grad_input, None


def sparsemax(input, dim = 0):
    return SparsemaxFunction.apply(input, dim)

class Sparsemax(nn.Module):

    def __init__(self, dim=0):
        self.dim = dim
        super(Sparsemax, self).__init__()

    def forward(self, input):
        return sparsemax(input, self.dim)


class LogSparsemax(nn.Module):

    def __init__(self, dim=0):
        self.dim = dim
        super(LogSparsemax, self).__init__()

    def forward(self, input):
        return torch.log(sparsemax(input, self.dim))

class SparsemaxLossFunction(Function):

    @staticmethod
    def forward(ctx, input, target):
        """
        input (FloatTensor): ``(n, num_classes)``.
        target (LongTensor): ``(n,)``, the indices of the target classes
        """
        input_batch, classes = input.size()
        target_batch = target.size(0)
        aeq(input_batch, target_batch)

        z_k = input.gather(1, target.unsqueeze(1)).squeeze()
        tau_z, support_size = _threshold_and_support(input, dim=1)
        support = input > tau_z
        x = torch.where(
            support, input**2 - tau_z**2,
            torch.tensor(0.0, device=input.device)
        ).sum(dim=1)
        ctx.save_for_backward(input, target, tau_z)
        # clamping necessary because of numerical errors: loss should be lower
        # bounded by zero, but negative values near zero are possible without
        # the clamp
        return torch.clamp(x / 2 - z_k + 0.5, min=0.0)

    @staticmethod
    def backward(ctx, grad_output):
        input, target, tau_z = ctx.saved_tensors
        sparsemax_out = torch.clamp(input - tau_z, min=0)
        delta = torch.zeros_like(sparsemax_out)
        delta.scatter_(1, target.unsqueeze(1), 1)
        return sparsemax_out - delta, None


def sparsemax_loss(input, target):
    return SparsemaxLossFunction.apply(input, target)

class SparsemaxLoss(nn.Module):
    """
    An implementation of sparsemax loss, first proposed in
    :cite:`DBLP:journals/corr/MartinsA16`. If using
    a sparse output layer, it is not possible to use negative log likelihood
    because the loss is infinite in the case the target is assigned zero
    probability. Inputs to SparsemaxLoss are arbitrary dense real-valued
    vectors (like in nn.CrossEntropyLoss), not probability vectors (like in
    nn.NLLLoss).
    """

    def __init__(self, weight=None, ignore_index=-100,
                 reduction='elementwise_mean'):
        assert reduction in ['elementwise_mean', 'sum', 'none']
        self.reduction = reduction
        self.weight = weight
        self.ignore_index = ignore_index
        super(SparsemaxLoss, self).__init__()

    def forward(self, input, target):
        loss = sparsemax_loss(input, target)
        if self.ignore_index >= 0:
            ignored_positions = target == self.ignore_index
            size = float((target.size(0) - ignored_positions.sum()).item())
            loss.masked_fill_(ignored_positions, 0.0)
        else:
            size = float(target.size(0))
        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'elementwise_mean':
            loss = loss.sum() / size
        return loss

# Masked multi-label versions.

class MultilabelSparsemaxLossFunction(Function):

    # assumes that target is 0 where mask is 0.
    @staticmethod
    def forward(ctx, input, target, mask = None):
        """
        input (FloatTensor): ``(n, num_classes)``.
        target (LongTensor): ``(n, num_classes)``, the probability dist over target classes
        """
        input_batch, classes = input.size()
        target_batch = target.size(0)
        aeq(input_batch, target_batch)

        qz = torch.einsum("ij,ij->i", [input, target])
        qnorm = torch.einsum("ij,ij->i", [target, target]) / 2.0

        if mask is not None:
            masked_input = input + mask.float().log()
        else:
            masked_input = input
        tau_z, support_size = _threshold_and_support(masked_input, dim=1)
        support = masked_input > tau_z
        x = torch.where(
            support, input**2 - tau_z**2,
            torch.tensor(0.0, device=input.device)
        ).sum(dim=1)
        ctx.save_for_backward(masked_input, target, tau_z)
        # clamping necessary because of numerical errors: loss should be lower
        # bounded by zero, but negative values near zero are possible without
        # the clamp
        return torch.clamp(x / 2 - qz + qnorm, min=0.0)

    @staticmethod
    def backward(ctx, grad_output):
        input, target, tau_z = ctx.saved_tensors
        sparsemax_out = torch.clamp(input - tau_z, min=0)
        return sparsemax_out - target, None, None


def multilabel_sparsemax_loss(input, target, mask = None):
    return MultilabelSparsemaxLossFunction.apply(input, target, mask)

class MultilabelSparsemaxLoss(nn.Module):

    def __init__(self, weight=None, ignore_index=-100,
                 reduction='elementwise_mean'):
        assert reduction in ['elementwise_mean', 'sum', 'none']
        self.reduction = reduction
        self.weight = weight
        self.ignore_index = ignore_index
        super(SparsemaxLoss, self).__init__()

    def forward(self, input, target):
        loss = multilabel_sparsemax_loss(input, target)
        if self.ignore_index >= 0:
            ignored_positions = target == self.ignore_index
            size = float((target.size(0) - ignored_positions.sum()).item())
            loss.masked_fill_(ignored_positions, 0.0)
        else:
            size = float(target.size(0))
        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'elementwise_mean':
            loss = loss.sum() / size
        return loss
