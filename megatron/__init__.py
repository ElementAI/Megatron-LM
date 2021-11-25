# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import typing

import torch
import os

logger = logging.getLogger(__name__)

from .package_info import (
    __description__,
    __contact_names__,
    __url__,
    __download_url__,
    __keywords__,
    __license__,
    __package_name__,
    __version__,
)

if "MEGATRON_SETUP" not in os.environ:
    from .global_vars import get_args
    from .global_vars import get_current_global_batch_size
    from .global_vars import get_num_microbatches
    from .global_vars import update_num_microbatches
    from .global_vars import get_tokenizer
    from .global_vars import get_tensorboard_writer
    from .global_vars import get_adlr_autoresume
    from .global_vars import get_timers
    from .initialize  import initialize_megatron

def print_rank_0(message):
    logger.info(str(message))

def is_last_rank():
    return torch.distributed.get_rank() == (
        torch.distributed.get_world_size() - 1)

def print_rank_last(message):
    logger.info(str(message))

_iteration=0
_metrics={}

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
    return metrics

def _log_dicts(self, metrics, indent=0):
    for key, value in metrics.items():
        key_ = key.rjust(len(key) + indent)
        # Merge keys when there is only one entry.
        while isinstance(value, dict) and len(value) == 1:
            for value_key, value_ in value.items():
                key_ = ".".join([key_, value_key])
                value = value_
        if isinstance(value, dict):
            logger.info(key_ + ":")
            self._log_dicts(value, indent + 2)
        else:
            sep = self._config.logging_width - len(value) - len(key_) - 2
            logger.info(f"{key_.ljust(len(key_)+sep,'.')}  {value}")
