import logging
import math

import torch
from megatron.global_vars import get_args

logger = logging.getLogger(__name__)

_iteration=0
_metrics={}
_LOGGING_WIDTH=50

def next_iteration(iteration:int):
    global _iteration, _metrics
    _metrics={}
    _iteration=iteration


def record_scale(name:str,x:torch.Tensor,grad=True, bias=None):
    global _metrics
    if get_log_scales():
        _metrics[f"{name}.scale" if grad else name]=get_scale(x if bias is None else x+bias)
        if grad and x.requires_grad:
            x.register_hook(lambda g: record_scale(f"{name}.grad",g,False))


def get_scale(x):
    return x.detach().float().pow(2).mean().pow(0.5)


def get_log_scales():
    args=get_args()
    return args.log_scales and (_iteration+1) % args.log_interval == 0


def log_metrics():
    metrics = {}
    for key, value in _metrics.items():
        metrics_ = metrics
        keys = key.split(".")
        for prefix in keys[:-1]:
            if prefix not in metrics_:
                metrics_[prefix] = {}
            metrics_ = metrics_[prefix]
        metrics_[keys[-1]] = _format_value(value)
    _log_dicts(metrics)


def _log_dicts(metrics, indent=0):
    for key, value in metrics.items():
        key_ = key.rjust(len(key) + indent)

        # Merge keys when there is only one entry.
        while isinstance(value, dict) and len(value) == 1:
            for value_key, value_ in value.items():
                key_ = ".".join([key_, value_key])
                value = value_
        if isinstance(value, dict):
            logger.info(key_ + ":")
            _log_dicts(value, indent + 2)
        else:
            sep = _LOGGING_WIDTH - len(value) - len(key_) - 2
            logger.info(f"{key_.ljust(len(key_)+sep,'.')}  {value}")


def _format_value(value, precision=5,max_leading_zeros=3):
    decimals = 0 if value == 0 or not math.isfinite(value) else precision - math.floor(math.log10(abs(value)))

    if 0 <= decimals <= precision + max_leading_zeros:
        value = f"{value:.{decimals}f}"
    else:
        value = f"{value:.{precision}e}"
    return value