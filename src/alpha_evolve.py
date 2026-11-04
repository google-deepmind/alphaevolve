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

"""A minimal implementation of AlphaEvolve with local evaluation."""

import asyncio
import traceback

from components import configuration
from components import evaluator_client
from components import evolution_database
from components import exceptions
from components import llm_client
from components import logging
from components import parsing_utils
from components import prompting

# Registries mapping configuration Enums to modules.
_DATABASES = {
    configuration.DatabaseType.GREEDY: evolution_database.GreedyDatabase,
    configuration.DatabaseType.GREEDY_AND_ISLANDS: (
        evolution_database.GreedyAndIslandsDatabase
    ),
}
_PROMPT_SAMPLERS = {
    configuration.PromptSamplerType.TEMPLATE: prompting.TemplatePromptSampler,
}
_LLM_CLIENTS = {
    configuration.LLMType.GEN_AI: llm_client.GenAILLMClient,
}
_EVALUATORS = {
    configuration.EvaluatorType.DOCKER: evaluator_client.DockerEvaluator,
}
_LOGGERS = {
    configuration.LoggerType.LOCAL: logging.LocalLogger,
}


async def step_loop(
    database: evolution_database.BaseDatabase,
    prompt_sampler: prompting.BasePromptSampler,
    llm: llm_client.BaseLLMClient,
    evaluator: evaluator_client.BaseEvaluator,
    logger: logging.BaseLogger,
    problem_description: str,
    total_steps: int,
) -> None:
  """Step loop of AlphaEvolve."""
  while logger.total_steps < total_steps:
    root_individual, *previous_individuals = database.sample_programs()
    llm_sample, code = None, None
    prompt = prompt_sampler.sample_prompt(
        problem_description,
        root_individual,
        previous_individuals,
    )
    try:
      llm_sample = await llm.generate_async(prompt)
      code = parsing_utils.apply_diffs(root_individual.text, llm_sample)
      scores, output = await evaluator.run_async(code, root_individual.output)
    except (
        exceptions.CodeExecutionError,
        exceptions.LLMInferenceError,
        exceptions.DiffParsingError,
        TypeError,
        ValueError,
    ) as e:
      tb = traceback.format_exc()
      logger.log_error(code, e, tb, prompt, llm_sample)
      continue
    program = evolution_database.Program(
        text=code,
        scores=scores,
        output=output,
        sampled_from=root_individual.sampled_from,
        island_index=root_individual.island_index,
    )
    database.register_program(program)
    logger.log_step(program, prompt, llm_sample)


async def async_run(
    config: configuration.Configuration,
) -> evolution_database.Program:
  """Main function of AlphaEvolve."""

  # Initialize modules and run the initial program.
  prompt_sampler = _PROMPT_SAMPLERS[config.modules.prompt_sampler]()
  llm = _LLM_CLIENTS[config.modules.llm]()
  evaluator = _EVALUATORS[config.modules.evaluator]()
  logger = _LOGGERS[config.modules.logger]()

  initial_scores, output = evaluator.run(config.problem_spec.initial_program)

  # Initialize database.
  database_cls = _DATABASES[config.modules.database]
  database = database_cls(
      initial_population=[
          evolution_database.Program(
              text=config.problem_spec.initial_program,
              scores=initial_scores,
              output=output,
          )
      ],
      main_metric_name=config.problem_spec.main_metric_name,
  )

  # Run the main loop in several parallel coroutines.
  futures = []
  async with asyncio.TaskGroup() as tg:
    for _ in range(config.budget.num_coroutines):
      futures.append(
          tg.create_task(
              step_loop(
                  database=database,
                  prompt_sampler=prompt_sampler,
                  llm=llm,
                  evaluator=evaluator,
                  logger=logger,
                  problem_description=config.problem_spec.problem_description,
                  total_steps=config.budget.num_llm_calls,
              )
          )
      )
    _, pending = await asyncio.wait(
        futures, return_when=asyncio.FIRST_COMPLETED
    )
    # Cancel the pending tasks. We want to terminate the experiment if any
    # coroutine completes. Waiting for all coroutines to complete can take a
    # long time, since some threads can get stuck and become unresponsive.
    for task in pending:
      task.cancel()

  logger.close()
  return database.get_best_program()
