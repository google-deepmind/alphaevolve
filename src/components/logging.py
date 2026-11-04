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

"""Logging utils for AlphaEvolve."""

import abc
import collections
from collections.abc import Mapping
import textwrap
from typing import Any

from components import evolution_database


class BaseLogger(abc.ABC):
  """Base class for logging programs and scores."""

  def __init__(self):
    self._step_stats = collections.defaultdict(int)
    self._error_stats = collections.defaultdict(int)
    self._best_scores = collections.defaultdict(lambda: float('-inf'))

  @property
  def total_steps(self) -> int:
    return sum(self._step_stats.values())

  @property
  def step_stats(self) -> Mapping[str, int]:
    return dict(self._step_stats)

  @property
  def error_stats(self) -> Mapping[str, int]:
    return dict(self._error_stats)

  def log_step(
      self,
      program: evolution_database.Program,
      prompt: str | None,
      llm_sample: str | None,
  ):
    """Logs a step and updates the best scores."""
    del prompt, llm_sample  # Unused.
    self._step_stats['successful_steps'] += 1
    # Update best scores:
    for metric, score in program.scores.items():
      metric_key = f'scores_to_optimize/{metric}'
      if score is not None and score > self._best_scores[metric_key]:
        self._best_scores[metric_key] = score

  def log_error(
      self,
      program: str | None,
      error: Exception,
      traceback: str | None,
      prompt: str | None,
      llm_sample: str | None,
  ):
    """Logs an error and updates the best scores."""
    del program, traceback, prompt, llm_sample  # Unused.
    self._step_stats['failed_steps'] += 1
    self._error_stats[type(error).__name__] += 1

  def close(self) -> None:
    """Closes the logger and flushes the logs."""
    # No action needed by default.
    pass

  def get_state(self) -> Mapping[str, Any]:
    return {
        'step_stats': dict(self._step_stats),
        'error_stats': dict(self._error_stats),
        'best_scores': dict(self._best_scores),
    }

  def set_state(self, state: Mapping[str, Any]) -> None:
    self._step_stats.update(state['step_stats'])
    self._error_stats.update(state['error_stats'])
    self._best_scores.update(state['best_scores'])


class LocalLogger(BaseLogger):
  """Logs programs and scores to the console."""

  def log_step(
      self,
      program: evolution_database.Program,
      prompt: str | None,
      llm_sample: str | None,
  ):
    """Logs a step and updates the best scores."""
    super().log_step(program, prompt, llm_sample)
    output = program.output
    output_str = (
        str(output) if len(str(output)) < 100 else f'{str(output)[:100]}...'
    )
    print(textwrap.dedent(f"""\
        --------------------------------------------------
        Step:        {self.total_steps}
        Program:     \n{textwrap.indent(str(program.text), ' ' * 8)}
        Output:      {output_str}
        Scores:      {program.scores}
        Best Scores: {dict(self._best_scores)}
        --------------------------------------------------
        """))

  def log_error(
      self,
      program: str | None,
      error: Exception,
      traceback: str | None,
      prompt: str | None,
      llm_sample: str | None,
  ):
    """Logs an error and updates the best scores."""
    super().log_error(program, error, traceback, prompt, llm_sample)
    print(f'step: {self.total_steps}\nerror: {error!r}\ntraceback: {traceback}')
