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

"""Configuration specifications for AlphaEvolve."""

import dataclasses
import enum
import yaml


class DatabaseType(str, enum.Enum):
  GREEDY = 'GreedyDatabase'
  GREEDY_AND_ISLANDS = 'GreedyAndIslandsDatabase'


class PromptSamplerType(str, enum.Enum):
  TEMPLATE = 'TemplatePromptSampler'


class LLMType(str, enum.Enum):
  GEN_AI = 'GenAILLMClient'


class EvaluatorType(str, enum.Enum):
  DOCKER = 'DockerEvaluator'


class LoggerType(str, enum.Enum):
  LOCAL = 'LocalLogger'


@dataclasses.dataclass(frozen=True, kw_only=True)
class ProblemSpec:
  """Specification of the problem to be optimized.

  Attributes:
    initial_program: The initial program to be evolved. This file is expected to
      contain some code enclosed in EVOLVE-BLOCK-* markers, as well as a
      function `evaluate` to run a single evaluation.
    problem_description: A description of the problem to be solved in natural
      language. This is appended to the prompt to the LLM for context.
    main_metric_name: The name of the main metric to be optimized. While
      AlphaEvolve optimizes all metrics returned by `evaluate`, this metric is
      the one that will be used to return the best-performing program at the end
      of the evolution.
  """

  initial_program: str
  problem_description: str
  main_metric_name: str


@dataclasses.dataclass(frozen=True, kw_only=True)
class Budget:
  """Budget for the evolution.

  Attributes:
    num_llm_calls: The total number of LLM calls allowed. This is used to set a
      bound on the total number of steps for the evolution process.
    num_coroutines: The number of coroutines to use for the evolution. This
      determines the degree of parallelism.
  """

  num_llm_calls: int
  num_coroutines: int


@dataclasses.dataclass(frozen=True, kw_only=True)
class Modules:
  """Modules used by the algorithm.

  This is to specify which modules are used by the main algorithm loop.
  AlphaEvolve is set up in a modular way, so it easily supports swapping out
  different modules (e.g. different LLM APIs, evaluators, ...).

  Attributes:
    database: The database module to use.
    prompt_sampler: The prompt sampler module to use.
    llm: The LLM module to use.
    evaluator: The evaluator module to use.
    logger: The logger module to use.
  """

  database: DatabaseType
  prompt_sampler: PromptSamplerType
  llm: LLMType
  evaluator: EvaluatorType
  logger: LoggerType


@dataclasses.dataclass(frozen=True, kw_only=True)
class Configuration:
  """Configuration for AlphaEvolve.

  Attributes:
    name: The name of the configuration.
    problem_spec: The specification of the problem to be optimized.
    budget: The budget for the evolution.
    modules: The modules used by the algorithm.
  """

  name: str
  problem_spec: ProblemSpec
  budget: Budget
  modules: Modules


def from_yaml(yaml_config: str) -> Configuration:
  """Loads a configuration from a YAML file."""
  config = yaml.safe_load(yaml_config)
  with open(config['problem_spec'].pop('program_path'), mode='r') as f:
    initial_program = f.read()
  config['problem_spec']['initial_program'] = initial_program

  modules_conf = config['modules']
  modules = Modules(
      database=DatabaseType(modules_conf['database']),
      prompt_sampler=PromptSamplerType(modules_conf['prompt_sampler']),
      llm=LLMType(modules_conf['llm']),
      evaluator=EvaluatorType(modules_conf['evaluator']),
      logger=LoggerType(modules_conf['logger']),
  )

  return Configuration(
      name=config['name'],
      problem_spec=ProblemSpec(**config['problem_spec']),
      budget=Budget(**config['budget']),
      modules=modules,
  )
