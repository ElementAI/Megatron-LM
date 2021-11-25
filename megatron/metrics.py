import logging

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


def record_scale(name:str,x:torch.Tensor,grad=True):
    global _metrics
    if get_log_scales():
        _metrics[name]=get_scale(x)
        if grad and x.requires_grad:
            x.register_hook(lambda g: record_scale(f"{name}_grad",g,False))


def get_scale(x):
    return x.detach().float().pow(2).mean().pow(0.5)


def get_log_scales():
    args=get_args()
    return args.log_scales and args.iteration % args.log_interval == 0


def log_metrics():
    metrics = {}
    for key, value in _metrics.items():
        metrics_ = metrics
        keys = key.split(".")
        for prefix in keys[:-1]:
            if prefix not in metrics_:
                metrics_[prefix] = {}
            metrics_ = metrics_[prefix]
        metrics_[keys[-1]] = value
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
