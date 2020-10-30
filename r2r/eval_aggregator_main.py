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

"""Evaluation aggregator for R2R."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from valan.framework import eval_aggregator
from valan.framework import hyperparam_flags  

FLAGS = flags.FLAGS


def main(_):
  eval_aggregator.run(
      aggregator_prefix=FLAGS.aggregator_prefix,
      logdir=FLAGS.logdir,
      server_address=FLAGS.server_address)


if __name__ == '__main__':
  app.run(main)
