import functools
import hashlib
import inspect
import json
import math
import os
import sys
import time
from argparse import Namespace
from collections import OrderedDict
from matplotlib import transforms
from matplotlib.patches import Ellipse
from types import MappingProxyType
from typing import *

import numpy as np
import torch
import torch.nn as nn
from dimarray import DimArray, Dataset
from tensordict import TensorDict
from torch.utils._pytree import tree_flatten, tree_unflatten

from infrastructure.settings import DEVICE


_T = TypeVar("_T")
"""
System and model functions
"""
def stack_tensor_arr(tensor_arr: np.ndarray[torch.Tensor], dim: int = 0) -> Union[torch.Tensor, TensorDict[str, torch.Tensor]]:
    tensor_list = [*tensor_arr.ravel()]
    if isinstance(t := tensor_list[0], torch.Tensor):
        result = torch.stack(tensor_list, dim=dim)
    else:
        result = TensorDict.maybe_dense_stack(tensor_list, dim=dim)
    return result.reshape(*tensor_arr.shape, *t.shape)


def stack_module_arr(module_arr: np.ndarray[nn.Module]) -> Tuple[nn.Module, TensorDict[str, torch.Tensor]]:
    params, buffers = torch.func.stack_module_state(module_arr.ravel().tolist())
    td = TensorDict({}, batch_size=module_arr.shape)

    def _unflatten(t: torch.Tensor, dim: int, shape: Tuple[int, ...]):
        if len(shape) == 0:
            return t.squeeze(dim=dim)
        elif len(shape) == 1:
            return t
        else:
            return t.unflatten(dim, shape)

    for k, v in params.items():
        td[(*k.split("."),)] = nn.Parameter(_unflatten(v, 0, module_arr.shape), requires_grad=v.requires_grad)
    for k, v in buffers.items():
        td[(*k.split("."),)] = _unflatten(v, 0, module_arr.shape)

    return module_arr.ravel()[0].to(DEVICE), td.to(DEVICE)


def stack_module_arr_preserve_reference(module_arr: np.ndarray[nn.Module]) -> Tuple[nn.Module, TensorDict[str, torch.Tensor]]:
    flattened_td = TensorDict.maybe_dense_stack([
        TensorDict({
            k: v
            for k in dir(module) if isinstance((v := getattr(module, k)), torch.Tensor)
        }, batch_size=())
        for module in module_arr.ravel()
    ], dim=0)
    td = flattened_td.reshape(module_arr.shape)
    return module_arr.ravel()[0], td.to(DEVICE)


def run_module_arr(
        reference_module: nn.Module,
        module_td: TensorDict[str, torch.Tensor],
        args: Any,  # Note: a TensorDict is only checked for as the immediate argument and will not work inside a nested structure
        kwargs: Dict[str, Any] = MappingProxyType(dict())
) -> Dict[str, Dict[str, torch.Tensor]]:
    if "TensorDict" in type(args).__name__:
        args = args.to_dict()

    module_td = TensorDict({
        k if isinstance(k, str) else ".".join(k): v
        for k, v in module_td.items(include_nested=True, leaves_only=True)
    }, batch_size=module_td.shape)

    try:
        def vmap_run(module_d, ags):
            return nn.utils.stateless.functional_call(reference_module, module_d, ags, kwargs)
        for _ in range(module_td.ndim):
            vmap_run = torch.func.vmap(vmap_run, randomness="different")
        return vmap_run(module_td.to_dict(), args)
    except RuntimeError:
        n = np.prod(module_td.shape)
        flat_args, args_spec = tree_flatten(args)
        single_flat_args_list = [
            [t.view(n, *t.shape[module_td.ndim:])[idx] for t in flat_args]
            for idx in range(n)
        ]
        single_args_list = [tree_unflatten(single_flat_args, args_spec) for single_flat_args in single_flat_args_list]

        single_out_list = [
            nn.utils.stateless.functional_call(reference_module, module_td.view(n)[idx].to_dict(), single_args)
            for idx, single_args in enumerate(single_args_list)
        ]
        _, out_spec = tree_flatten(single_out_list[0])
        single_flat_out_list = [tree_flatten(single_out)[0] for single_out in single_out_list]
        flat_out = [
            torch.stack([*out_component_list], dim=0).view(*module_td.shape, *out_component_list[0].shape)
            for out_component_list in zip(*single_flat_out_list)
        ]
        return tree_unflatten(flat_out, out_spec)


def double_vmap(func: Callable) -> Callable:
    return torch.vmap(torch.vmap(func))


def buffer_dict(td: TensorDict[str, torch.Tensor]) -> nn.Module:
    def _buffer_dict(parent_module: nn.Module, td: TensorDict[str, torch.Tensor]) -> nn.Module:
        for k, v in td.items(include_nested=False):
            if isinstance(v, torch.Tensor):
                parent_module.register_buffer(k, v)
            else:
                parent_module.register_module(k, _buffer_dict(nn.Module(), v))
        return parent_module
    return _buffer_dict(nn.Module(), td)


def mask_dataset_with_total_sequence_length(ds: TensorDict[str, torch.Tensor], total_sequence_length: int) -> TensorDict[str, torch.Tensor]:
    batch_size, sequence_length = ds.shape[-2:]
    ds["mask"] = torch.Tensor(torch.arange(batch_size * sequence_length) < total_sequence_length).view(
        sequence_length, batch_size
    ).mT.expand(ds.shape)
    return ds


"""
Computation
"""
def pow_series(M: torch.Tensor, n: int) -> torch.Tensor:
    N = M.shape[0]
    I = torch.eye(N, device=M.device)
    if n == 1:
        return I[None]
    else:
        k = int(math.ceil(math.log2(n)))
        bits = [M]
        for _ in range(k - 1):
            bits.append(bits[-1] @ bits[-1])

        result = I
        for bit in bits:
            augmented_bit = torch.cat([I, bit], dim=1)
            blocked_result = result @ augmented_bit
            result = torch.cat([blocked_result[:, :N], blocked_result[:, N:]], dim=0)
        return result.reshape(1 << k, N, N)[:n]


def batch_trace(x: torch.Tensor) -> torch.Tensor:
    return x.diagonal(dim1=-2, dim2=-1).sum(dim=-1)


def kl_div(cov1: torch.Tensor, cov2: torch.Tensor) -> torch.Tensor:
    return ((torch.det(cov2) / torch.det(cov1)).log() - cov1.shape[-1] + (torch.inverse(cov2) * cov1).sum(dim=(-2, -1))) / 2


def sqrtm(t: torch.Tensor) -> torch.Tensor:
    L, V = torch.linalg.eig(t)
    return (V @ torch.diag_embed(L ** 0.5) @ torch.inverse(V)).real


def complex(t: torch.Tensor | TensorDict[str, torch.Tensor]) -> Union[torch.Tensor, TensorDict[str, torch.Tensor]]:
    fn = lambda t_: torch.complex(t_, torch.zeros_like(t_))
    return fn(t) if isinstance(t, torch.Tensor) else t.apply(fn)


def ceildiv(a: int, b: int) -> int:
    return -(-a // b)


def prod(a: Union[int, float]) -> Union[int, float]:
    return np.prod(a).item()


def multiclass_logits(t: torch.Tensor) -> torch.Tensor:
    logits = torch.log(t)
    return logits - torch.mean(logits, dim=-1, keepdim=True)


def hadamard_conjugation(
        A: torch.Tensor,        # [B... x m x n]
        B: torch.Tensor,        # [B... x p x q]
        alpha: torch.Tensor,    # [B... x m x n]
        beta: torch.Tensor,     # [B... x p x q]
        C: torch.Tensor         # [B... x m x p]
) -> torch.Tensor:              # [B... x n x q]
    P = A[..., :, None, :, None] * B[..., None, :, None, :]                         # [B... x m x p x n x q]
    coeff = 1 / (1 - alpha[..., :, None, :, None] * beta[..., None, :, None, :])    # [B... x m x p x n x q]
    return torch.sum(P * coeff * C[..., None, None], dim=[-3, -4])


def hadamard_conjugation_diff_order1(
        A: torch.Tensor,        # [B... x m x n]
        B: torch.Tensor,        # [B... x p x q]
        alpha: torch.Tensor,    # [B... x m x n]
        beta1: torch.Tensor,    # [B... x p x q]
        beta2: torch.Tensor,    # [B... x p x q]
        C: torch.Tensor         # [B... x m x p]
) -> torch.Tensor:              # [B... x n x q]
    P = A[..., :, None, :, None] * B[..., None, :, None, :]
    alpha_ = alpha[..., :, None, :, None]
    _beta1, _beta2 = beta1[..., None, :, None, :], beta2[..., None, :, None, :]
    coeff = alpha_ / ((1 - alpha_ * _beta1) * (1 - alpha_ * _beta2))
    return torch.sum(P * coeff * C[..., None, None], dim=[-3, -4])


def hadamard_conjugation_diff_order2(
        B: torch.Tensor,        # [B... x p x q]
        beta1: torch.Tensor,    # [B... x p x q]
        beta2: torch.Tensor,    # [B... x p x q]
        C: torch.Tensor         # [B... x p x p]
) -> torch.Tensor:              # [B... x q x q]
    P = B[..., :, None, :, None] * B[..., None, :, None, :]                         # [B... x p x p x q x q]
    beta1_, _beta1 = beta1[..., :, None, :, None], beta1[..., None, :, None, :]     # b1_ik, b1_jl
    beta2_, _beta2 = beta2[..., :, None, :, None], beta2[..., None, :, None, :]     # b2_ik, b2_jl

    beta12 = beta1_ * _beta2                                                        # b1_ik * b2_jl
    beta21 = beta12.transpose(dim0=-4, dim1=-3).transpose(dim0=-2, dim1=-1)         # b2_ik * b1_jl

    coeff = (1 - beta12 * beta21) / (                                               # 1 - (b1 * b2)_ik * (b1 * b2)_jl
        (1 - beta1_ * _beta1)                                                       # 1 - b1_ik * b1_jl
        * (1 - beta12)                                                              # 1 - b1_ik * b2_jl
        * (1 - beta21)                                                              # 1 - b2_ik * b1_jl
        * (1 - beta2_ * _beta2)                                                     # 1 - b2_ik * b2_jl
    )
    return torch.sum(P * coeff * C[..., None, None], dim=[-3, -4])


class InverseCubic(nn.Module):
    class _InverseCubic(torch.autograd.Function):
        @staticmethod
        def forward(ctx, t: torch.Tensor) -> torch.Tensor:
            c = t * 2.598076211353
            k = torch.pow(torch.sqrt(torch.square(c) + 1) + c, 0.333333333333)
            r = k - 1 / k
            ctx.save_for_backward(r)
            return 0.577350269190 * r

        @staticmethod
        def backward(ctx, t: torch.Tensor) -> torch.Tensor:
            r, = ctx.saved_tensors
            return t / (torch.square(r) + 1)

    def __init__(self):
        super().__init__()
        self.op = InverseCubic._InverseCubic()

    def forward(self, t: torch.Tensor):
        return self.op.apply(t)

inverse_cubic = InverseCubic._InverseCubic.apply


"""
NumPy Array Comprehension Operations
"""
def multi_iter(arr: np.ndarray | DimArray) -> Iterable[Any]:
    for x in np.nditer(arr, flags=["refs_ok"]):
        yield x[()]


def multi_enumerate(arr: np.ndarray | DimArray) -> Iterable[Tuple[Sequence[int], Any]]:
    it = np.nditer(arr, flags=["multi_index", "refs_ok"])
    for x in it:
        yield it.multi_index, x[()]

def multi_map(func: Callable[[Any], Any], arr: np.ndarray | DimArray, dtype: type = None):
    if dtype is None:
        dtype = type(func(arr.ravel()[0]))
    result = np.empty_like(arr, dtype=dtype)
    for idx, x in multi_enumerate(arr):
        result[idx] = func(x)
    return DimArray(result, dims=arr.dims) if isinstance(arr, DimArray) else result

def multi_zip(*arrs: np.ndarray) -> np.ndarray:
    result = np.recarray(arrs[0].shape, dtype=[(f"f{i}", arr.dtype) for i, arr in enumerate(arrs)])
    for i, arr in enumerate(arrs):
        setattr(result, f"f{i}", arr)
    return result


"""
DimArray Operations
"""

def dim_array_like(arr: DimArray, dtype: type) -> DimArray:
    empty_arr = np.full_like(arr, None, dtype=dtype)
    return DimArray(empty_arr, dims=arr.dims)

def broadcast_dim_array_shapes(*dim_arrs: Iterable[DimArray]) -> OrderedDict[str, int]:
    dim_dict = OrderedDict()
    for dim_arr in dim_arrs:
        for dim_name, dim_len in zip(dim_arr.dims, dim_arr.shape):
            dim_dict.setdefault(dim_name, []).append(dim_len)
    return OrderedDict((k, np.broadcast_shapes(*v)[0]) for k, v in dim_dict.items())

def broadcast_dim_arrays(*dim_arrs: Iterable[np.ndarray]) -> Iterator[DimArray]:
    _dim_arrs = []
    for dim_arr in dim_arrs:
        if isinstance(dim_arr, DimArray):
            _dim_arrs.append(dim_arr)
        elif isinstance(dim_arr, np.ndarray):
            assert dim_arr.ndim == 0
            _dim_arrs.append(DimArray(dim_arr, dims=[]))
        else:
            _dim_arrs.append(DimArray(array_of(dim_arr), dims=[]))
    dim_arrs = _dim_arrs

    dim_dict = broadcast_dim_array_shapes(*dim_arrs)
    reference_dim_arr = DimArray(
        np.zeros((*dim_dict.values(),)),
        dims=(*dim_dict.keys(),),
        axes=(*map(np.arange, dim_dict.values()),)
    )
    return (dim_arr.broadcast(reference_dim_arr) for dim_arr in dim_arrs)

def take_from_dim_array(dim_arr: DimArray | Dataset, idx: Dict[str, Any]):
    dims = set(dim_arr.dims)
    return dim_arr.take(indices={k: v for k, v in idx.items() if k in dims})


"""
Recursive attribute functions
"""
def rgetattr(obj: object, attr: str, *args):
    def _getattr(obj: object, attr: str) -> Any:
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split("."))


def rsetattr(obj: object, attr: str, value: Any) -> None:
    def _rsetattr(obj: object, attrs: List[str], value: Any) -> None:
        if len(attrs) == 1:
            setattr(obj, attrs[0], value)
        else:
            _rsetattr(next_obj := getattr(obj, attrs[0], Namespace()), attrs[1:], value)
            setattr(obj, attrs[0], next_obj)
    _rsetattr(obj, attr.split("."), value)


def rhasattr(obj: object, attr: str) -> bool:
    try:
        rgetattr(obj, attr)
        return True
    except AttributeError:
        return False


def rgetitem(obj: Dict[str, Any], item: str, *args):
    def _getitem(obj: Dict[str, Any], item: str) -> Any:
        return obj.get(item, *args)
    return functools.reduce(_getitem, [obj] + item.split("."))


"""
Argument namespace processing
"""
def deepcopy_namespace(n: Namespace) -> Namespace:
    def _deepcopy_helper(o: _T) -> _T:
        if isinstance(o, Namespace):
            return type(o)(**{k: _deepcopy_helper(v) for k, v in vars(o).items()})
        else:
            return o
    return _deepcopy_helper(n)


def toJSON(o: object):
    if isinstance(o, Namespace):
        return {k: toJSON(v) for k, v in vars(o).items()}
    elif isinstance(o, dict):
        return {k: toJSON(v) for k, v in o.items()}
    elif isinstance(o, (list, tuple, set)):
        return list(map(toJSON, o))
    else:
        try:
            json.dumps(o)
            return o
        except TypeError:
            return str(o)


def str_namespace(n: Namespace) -> str:
    return json.dumps(toJSON(n), indent=4)


def print_namespace(n: Namespace) -> None:
    print(str_namespace(n))


def hash_namespace(n: Namespace) -> str:
    return hashlib.sha256(str_namespace(n).encode("utf-8")).hexdigest()[:8]


"""
Miscellaneous
"""
class PTR(object):
    def __init__(self, obj: object) -> None:
        self.obj = obj

    def __iter__(self):
        yield self.obj


class print_disabled:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class Timer:
    def __init__(self):
        self.start = time.perf_counter()
        
    def stop(self):
        stop = time.perf_counter()
        result = stop - self.start
        self.start = stop
        return result


def flatten_nested_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    result = {}
    def _flatten_nested_dict(s: Tuple[str, ...], d: Dict[str, Any]) -> None:
        for k, v in d.items():
            if isinstance(v, dict):
                _flatten_nested_dict((*s, k), v)
            else:
                result[".".join((*s, k))] = v
    _flatten_nested_dict((), d)
    return result


def nested_vars(n: Namespace) -> Dict[str, Any]:
    result = {}
    def _nested_vars(s: Tuple[str, ...], n: Namespace) -> None:
        for k, v in vars(n).items():
            if isinstance(v, Namespace):
                _nested_vars((*s, k), v)
            else:
                result[(*s, k)] = v
    _nested_vars((), n)
    return {".".join(k): v for k, v in result.items()}


def nested_type(o: object) -> object:
    if type(o) in [list, tuple]:
        return type(o)(map(nested_type, o))
    elif type(o) == dict:
        return {k: nested_type(v) for k, v in o.items()}
    else:
        return type(o)


def map_dict(d: Dict[str, Any], func: Callable[[Any], Any]) -> Dict[str, Any]:
    return {
        k: map_dict(v, func) if hasattr(v, "items") else func(v)
        for k, v in d.items()
    }


def array_of(o: _T) -> np.ndarray[_T]:
    M = np.array(None, dtype=object)
    M[()] = o
    return M


def model_size(m: nn.Module):
    return sum(p.numel() for p in m.parameters())


def call_func_with_kwargs(func: Callable, args: Tuple[Any, ...], kwargs: Dict[str, Any]):
    kwargs = {**kwargs}
    while True:
        try:
            func(*args, **kwargs)
            break
        except AttributeError as excp:
            del kwargs[str(excp).split("'")[1]]

    # params = inspect.signature(func).parameters
    # required_args = [
    #     kwargs[k] if k in kwargs else args[i] for i, (k, v) in enumerate(params.items())
    #     if v.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD and v.default is inspect.Parameter.empty
    # ]
    # additional_args = args[len(required_args):]
    #
    # allow_var_keywords = any(v.kind is inspect.Parameter.VAR_KEYWORD for v in params.values())
    # valid_kwargs = {
    #     k: v for k, v in kwargs.items()
    #     if ((params[k].default is not inspect.Parameter.empty) if k in params else allow_var_keywords)
    # }
    # return func(*required_args, *additional_args, **valid_kwargs)


""" Plotting code """
def color(z: float, scale: float = 120.) -> np.ndarray:
    k = 2 * np.pi * z / scale
    return (1 + np.asarray([np.sin(k), np.sin(k + 2 * np.pi / 3), np.sin(k + 4 * np.pi / 3)], dtype=float)) / 2


def confidence_ellipse(x, y, ax, n_std=1.0, facecolor="none", **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The Axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse"s radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    x, y = np.array(x), np.array(y)
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    M = np.stack([x, y], axis=0)
    cov = (M @ M.T) / len(x)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from the squareroot of the variance and multiplying with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std

    # Calculating the standard deviation of y
    scale_y = np.sqrt(cov[1, 1]) * n_std

    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)




