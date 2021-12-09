"""
Author: Adrish Dey (adrish@wandb.com)
Dataset Loader Dispatcher For Pytorch and TensorFlow Backends
"""
import enum
import functools
import importlib

import numpy as np


class Environment(enum.Enum):
    TORCH = 1


def cache_fn():
    """
    Output Caching for Function Calls.
    Used for Speeding Up IO operations.

    DISCLAIMER: The Cache has no upper limit.
    For large datasets / limited RAM
    use functools.lru_cache instead
    """
    def cache_wrapper(function):
      cache = {}
      @functools.wraps(function)
      def wrapper_fn(*args):
          key = tuple(args)
          if key in cache:
              return cache[key]
          cache[key] = function(*args)
          return cache[key]
      return wrapper_fn
    return cache_wrapper


def load_dataset(path, transform_fn, cache_fn=functools.lru_cache()):
    """
    Load TACO dataset from a specified path
    for a given Environment
    Args:
        path(str): Path to Dataset Folder

        env(tacoloader.Environment): Enumerator for ML Library
        being used.

        transform_fn(`callable`): Preprocessing function that takes
        an image tensor and outputs an image tensor

        cache_fn(`callable`, default: functools.lru_cache()):
        Function for caching IO operations between epochs.

    """
    class_name = "tacoloader.%s_loader" % env.name.lower()
    module = importlib.import_module(class_name)
    return (
        module.TacoDataset(path, transform_fn, cache_fn),
        module.TacoDataset.collate_fn,
    )
