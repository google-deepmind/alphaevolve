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

"""Database for storing and sampling programs."""

import abc
from collections.abc import Mapping, Sequence
import dataclasses
import datetime
import random
import re
from typing import Any

from components import parsing_utils
import numpy as np
from typing_extensions import override


@dataclasses.dataclass(frozen=True, kw_only=True)
class Program:
  """A program stored in the database.

  Attributes:
    text: The program text, usually a Python code module.
    scores: The scores returned by executing the program. These are the metrics
      of the field `scores_to_maximize` from the `evaluate` function.
    output: The (optional) additional output of the program. This is the field
      `output_artifacts` from the `evaluate` function, if present.
    sampled_from: The strategy used to sample the program, indicating which part
      of the database the program was sampled from (island or best programs).
    island_index: The index of the island the program was sampled from. This is
      only set if `sampled_from` is 'islands'.
  """

  text: str
  scores: dict[str, float | None]
  output: Any | None = None
  sampled_from: str | None = None
  island_index: int | None = None

  def render(self) -> str:
    code = ''.join(
        re.findall(
            parsing_utils.evolve_block_pattern('(.*?)'), self.text, re.DOTALL
        )
    )
    return '\n'.join([
        f'Obtained scores: {self.scores}',
        parsing_utils.code_block_pattern(code),
    ])


class BaseDatabase(abc.ABC):
  """Base class for databases."""

  def __init__(
      self, initial_population: Sequence[Program], main_metric_name: str
  ):
    self._main_metric_name = main_metric_name
    self._initial_population = initial_population

  @abc.abstractmethod
  def register_program(self, program: Program) -> None:
    """Registers a new program if it improves any metric."""

  @abc.abstractmethod
  def sample_programs(self, k: int = 5) -> Sequence[Program]:
    """Samples k programs from the database."""

  @abc.abstractmethod
  def get_best_program(self) -> Program:
    """Returns the current best program."""

  def get_state(self) -> Mapping[str, Any]:
    return {}

  def set_state(self, state: Mapping[str, Any]) -> None:
    del state  # Unused.


class GreedyDatabase(BaseDatabase):
  """Database that stores the best programs for each metric."""

  def __init__(
      self, initial_population: Sequence[Program], main_metric_name: str
  ):
    super().__init__(
        initial_population=initial_population,
        main_metric_name=main_metric_name,
    )
    self._best_programs_per_metric: dict[str, Program] = {}
    for program in initial_population:
      self.register_program(program)

  @override
  def register_program(self, program: Program) -> None:
    """Registers a new program if it improves any metric."""
    for metric, score in program.scores.items():
      if score is None:
        continue
      if (
          metric not in self._best_programs_per_metric
          or score > self._best_programs_per_metric[metric].scores[metric]
      ):
        self._best_programs_per_metric[metric] = program

  @override
  def sample_programs(self, k: int = 5) -> Sequence[Program]:
    """Samples k programs from the database."""
    programs = list(self._best_programs_per_metric.values())
    unique_programs = list({p.text: p for p in programs}.values())
    return random.sample(unique_programs, min(k, len(unique_programs)))

  @override
  def get_best_program(self) -> Program:
    """Returns the best program."""
    return self._best_programs_per_metric[self._main_metric_name]

  def get_state(self) -> Mapping[str, Any]:
    return super().get_state() | {
        'best_programs_per_metric': self._best_programs_per_metric,
    }

  def set_state(self, state: Mapping[str, Any]) -> None:
    super().set_state(state)
    self._best_programs_per_metric = state['best_programs_per_metric']


class GreedyAndIslandsDatabase(BaseDatabase):
  """Database combining Greedy and Islands evolution strategies."""

  def __init__(
      self,
      initial_population: Sequence[Program],
      main_metric_name: str,
      num_islands: int = 5,
      island_reset_interval_hours: float = 2.0,
  ):
    super().__init__(
        initial_population=initial_population,
        main_metric_name=main_metric_name,
    )
    self._num_islands = num_islands
    self._island_reset_interval_hours = island_reset_interval_hours
    self._best_programs_per_metric: dict[str, Program] = {}
    self._islands: Sequence[list[Program]] = [[] for _ in range(num_islands)]
    self._last_reset_time = datetime.datetime.now()
    for program in initial_population:
      self._register_to_best_programs_per_metric(program)
      for island_index in range(num_islands):
        self._register_to_islands(program, island_index)

  @override
  def register_program(self, program: Program) -> None:
    self._register_to_best_programs_per_metric(program)
    if program.sampled_from == 'islands' and program.island_index is not None:
      self._register_to_islands(program, program.island_index)
    self._maybe_reset_islands()

  def _register_to_best_programs_per_metric(self, program: Program) -> None:
    for metric, score in program.scores.items():
      if score is None:
        continue
      current_best = self._best_programs_per_metric.get(metric)
      if current_best is None or score > current_best.scores[metric]:
        self._best_programs_per_metric[metric] = dataclasses.replace(
            program, sampled_from=None, island_index=None
        )

  def _register_to_islands(self, program: Program, island_idx: int) -> None:
    if 0 <= island_idx < len(self._islands):
      self._islands[island_idx].append(
          dataclasses.replace(program, sampled_from=None, island_index=None)
      )

  @override
  def sample_programs(self, k: int = 5) -> Sequence[Program]:
    if random.random() < 0.5:
      return self._sample_from_best_programs_per_metric(k)
    else:
      return self._sample_from_islands(k)

  def _sample_from_best_programs_per_metric(self, k: int) -> Sequence[Program]:
    programs = list(self._best_programs_per_metric.values())
    unique_programs = list({p.text: p for p in programs}.values())
    sampled_programs = random.sample(
        unique_programs, min(k, len(unique_programs))
    )
    return [
        dataclasses.replace(p, sampled_from='best_programs_per_metric')
        for p in sampled_programs
    ]

  def _sample_from_islands(self, k: int) -> Sequence[Program]:
    # Check that all islands are non-empty.
    if not all(self._islands):
      raise ValueError('Some islands do not contain any programs.')
    # Sample a random island and program from it.
    island_index = random.choice(range(len(self._islands)))
    island = self._islands[island_index]
    sampled_programs = self._temperature_sample(island, k)
    return [
        dataclasses.replace(
            p, sampled_from='islands', island_index=island_index
        )
        for p in sampled_programs
    ]

  def _temperature_sample(
      self, programs: Sequence[Program], k: int, temperature: float = 0.1
  ) -> Sequence[Program]:
    """Samples programs using temperature-scaled softmax."""
    if not programs:
      return []
    metric_name = random.choice(list(programs[0].scores.keys()))
    scores = np.array([p.scores.get(metric_name, -np.inf) for p in programs])
    if np.all(scores == -np.inf):
      probabilities = np.ones(len(programs)) / len(programs)
    else:
      finite_scores = scores[scores > -np.inf]
      min_score = np.min(finite_scores)
      normalized = np.where(scores > -np.inf, scores - min_score, -np.inf)
      normalized = normalized / (np.max(normalized) + 1e-6)
      logits = normalized / temperature
      logits_max = np.max(logits)
      exp_logits = np.exp(logits - logits_max)
      probabilities = exp_logits / np.sum(exp_logits) + 1e-6
      probabilities = probabilities / np.sum(probabilities)
    k = min(k, len(programs))
    indices = np.random.choice(
        len(programs), size=k, replace=False, p=probabilities
    )
    sampled_programs = [programs[i] for i in indices]
    return list({p.text: p for p in sampled_programs}.values())

  @override
  def get_best_program(self) -> Program:
    return self._best_programs_per_metric[self._main_metric_name]

  def _maybe_reset_islands(self) -> None:
    elapsed = datetime.datetime.now() - self._last_reset_time
    if elapsed.total_seconds() < self._island_reset_interval_hours * 3600:
      return
    self._reset_islands()
    self._last_reset_time = datetime.datetime.now()

  def _reset_islands(self, k: int = 5) -> None:
    """Resets the islands by reseeding them with top k programs per island."""
    seeds = []
    for island in self._islands:
      seeds.extend(self._temperature_sample(island, k, temperature=1e-6))
    self._islands: Sequence[list[Program]] = [
        [] for _ in range(self._num_islands)
    ]
    unique_seeds = list({p.text: p for p in seeds}.values())
    if len(unique_seeds) < len(self._islands):
      # If there are not enough unique seeds, repeat sampling from the
      # population until there are enough unique seeds.
      unique_seeds = random.sample(
          unique_seeds,
          len(self._islands),
          counts=[len(self._islands)] * len(unique_seeds),
      )
    for i, program in enumerate(unique_seeds):
      self._islands[i % self._num_islands].append(program)

  @override
  def get_state(self) -> Mapping[str, Any]:
    return super().get_state() | {
        'best_programs_per_metric': self._best_programs_per_metric,
        'islands': self._islands,
        'last_reset_time': self._last_reset_time,
    }

  @override
  def set_state(self, state: Mapping[str, Any]) -> None:
    super().set_state(state)
    self._best_programs_per_metric = state['best_programs_per_metric']
    self._islands = state['islands']
    self._last_reset_time = state['last_reset_time']
