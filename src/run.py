# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Run AlphaEvolve with local evaluation."""

import argparse
import asyncio
from collections.abc import Sequence
import logging
import os
import sys

import alpha_evolve
from components import configuration


def main(argv: Sequence[str]) -> None:
  parser = argparse.ArgumentParser(description='Run AlphaEvolve.')
  parser.add_argument(
      '--problem_config',
      type=str,
      required=True,
      help='Path to the .yaml problem config file.',
  )
  args = parser.parse_args(argv[1:])

  # Configure basic logging
  logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

  problem_config_path = args.problem_config
  if not os.path.exists(problem_config_path):
    logging.error('Problem config file not found: %s', problem_config_path)
    sys.exit(1)

  with open(problem_config_path, 'r') as f:
    problem_config_yaml = f.read()
  config = configuration.from_yaml(problem_config_yaml)

  best_program = asyncio.run(alpha_evolve.async_run(config))
  logging.info('Best program: %s', best_program.text)
  logging.info('Best scores: %s', best_program.scores)


if __name__ == '__main__':
  main(sys.argv)
