# coding=utf-8
# Copyright 2019 Google LLC
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

"""Class for hyperparameters."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import six


class HParams(object):
  """Class to hold a set of hyperparameters as name-value pairs."""

  _HAS_DYNAMIC_ATTRIBUTES = True  # Required for pytype checks.

  def __init__(self, **kwargs):
    """Create an instance of `HParams` from keyword arguments.

    The keyword arguments specify name-values pairs for the hyperparameters.
    The parameter types are inferred from the type of the values passed.

    The parameter names are added as attributes of `HParams` object, so they
    can be accessed directly with the dot notation `hparams._name_`.

    Example:

    ```python
    # Define 3 hyperparameters: 'learning_rate' is a float parameter,
    # 'num_hidden_units' an integer parameter, and 'activation' a string
    # parameter.
    hparams = HParams(learning_rate=0.1, num_hidden_units=100,
                      activation='relu')

    hparams.activation ==> 'relu'
    ```

    Note that a few names are reserved and cannot be used as hyperparameter
    names.  If you use one of the reserved name the constructor raises a
    `ValueError`.

    Args:
      **kwargs: Key-value pairs where the key is the hyperparameter name and the
        value is the value for the parameter.

    Raises:
      ValueError: If one of the arguments is invalid.

    """
    # Register the hyperparameters and their type in _hparam_types.
    # _hparam_types maps the parameter name to a tuple (type, bool).
    # The type value is the type of the parameter for scalar hyperparameters,
    # or the type of the list elements for multidimensional hyperparameters.
    # The bool value is True if the value is a list, False otherwise.
    self._hparam_types = {}
    for name, value in six.iteritems(kwargs):
      self.add_hparam(name, value)

  def add_hparam(self, name, value):
    """Adds {name, value} pair to hyperparameters.

    Args:
      name: Name of the hyperparameter.
      value: Value of the hyperparameter. Can be one of the following types:
        int, float, string, int list, float list, or string list.

    Raises:
      ValueError: if one of the arguments is invalid.
    """
    # Keys in kwargs are unique, but 'name' could be the name of a pre-existing
    # attribute of this object.
    if getattr(self, name, None) is not None:
      raise ValueError('Hyperparameter name is reserved: %s' % name)
    if isinstance(value, (list, tuple)):
      if not value:
        raise ValueError('Multi-valued hyperparameters cannot be empty: %s' %
                         name)
      self._hparam_types[name] = (type(value[0]), True)
    else:
      self._hparam_types[name] = (type(value), False)
    setattr(self, name, value)

  def values(self):
    """Return the hyperparameter values as a Python dictionary.

    Returns:
      A dictionary with hyperparameter names as keys.  The values are the
      hyperparameter values.
    """
    return {n: getattr(self, n) for n in self._hparam_types.keys()}
