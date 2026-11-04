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

"""Module that samples a mutation prompt."""

import abc
from collections.abc import Sequence
import json
import random

from components import evolution_database
from components import parsing_utils
from typing_extensions import override


class BasePromptSampler(abc.ABC):
  """Base class for sampling a mutation prompt."""

  @abc.abstractmethod
  def sample_prompt(
      self,
      problem_description: str,
      root_individual: evolution_database.Program,
      previous_individuals: Sequence[evolution_database.Program],
  ) -> str:
    """Samples a mutation prompt."""


PROMPT_TEMPLATE = """
{problem_description}

# Prior programs
Previously we found that the following programs performed well on the task at hand:

{previous_programs}

# Current program
Here is the current program we are trying to improve (you will need to propose
a modification to it below):

{code}

# *SEARCH/REPLACE block* Rules:

Every *SEARCH/REPLACE block* must use this format:
1. The opening fence: ```python
2. The start of search block: <<<<<<< SEARCH
3. A contiguous chunk of up to 4 lines to search for in the existing source code
4. The dividing line: =======
5. The lines to replace into the source code
6. The end of the replace block: >>>>>>> REPLACE
7. The closing fence: ```

Every *SEARCH* section must *EXACTLY MATCH* the existing file content,
character for character, including all comments, docstrings, etc.

*SEARCH/REPLACE* blocks will replace *all* matching occurrences.
Include enough lines to make the SEARCH blocks uniquely match the lines to
change.

Keep *SEARCH/REPLACE* blocks concise.
Break large *SEARCH/REPLACE* blocks into a series of smaller blocks that each
change a small portion of the file.
Include just the changing lines, and a few surrounding lines if needed for
uniqueness.
Do not include long runs of unchanging lines in *SEARCH/REPLACE* blocks.

To move code within a file, use 2 *SEARCH/REPLACE* blocks: 1 to delete it from
its current location, 1 to insert it in the new location.

Make sure that the changes you propose are consistent with each other. For
example, if you refer to a new config variable somewhere, you should also
propose a change to add that variable.

Example:
```python
{diff_search}
    return total_loss
{diff_mid}
    # Add sparsity-promoting regularization to the loss.
    total_loss += self.hypers.l1_reg_weight * l1_reg

    return total_loss
{diff_replace}
```
and
```python
{diff_search}
  return hyper.zipit([
{diff_mid}
  return hyper.zipit([
      hyper.uniform('l1_reg_weight', hyper.interval(0.0, 0.01)),
{diff_replace}
```

ONLY EVER RETURN CODE IN A *SEARCH/REPLACE BLOCK*!

# Task
{task_instruction} {focus_sentence} {trigger_chain_of_thought}
Describe each change with a *SEARCH/REPLACE block*.
"""

PROMPT_FORMAT_ARGS = r"""
{
  "task_instruction": [
    ["Propose modifications to current program that combine the strengths of all the programs above and achieved high scores on the task.", 0.2],
    ["Suggest a crazy idea of how we can improve our implementation.", 0.2],
    ["Suggest a crazy idea of how we can improve our implementation, something that definitely nobody else would think of. Make it crazy with a capital C.", 0.2],
    ["Suggest a new idea to improve the code.", 0.2],
    ["Suggest a new idea to improve the code that is inspired by your expert knowledge of optimization.", 0.2]
  ],
  "focus_sentence": [
    ["Focus on simplifying the code, instead of adding new functionality.", 0.2],
    ["", 0.8]
  ],
  "trigger_chain_of_thought": [
    ["\n\nStart with providing a comprehensive explanation for the proposed changes including\n* The specific issue or limitation it addresses.\n* The underlying rationale and expected impact.\n\nYou need to specify this *before providing code*.\n\n", 0.5],
    ["\n\n", 0.5]
  ]
}
"""

_DIFF_PARAMS = {
    'diff_search': parsing_utils.SEARCH,
    'diff_mid': parsing_utils.DIVIDER,
    'diff_replace': parsing_utils.REPLACE,
}


class TemplatePromptSampler(BasePromptSampler):
  """Samples a mutation prompt using a string template."""

  @override
  def sample_prompt(
      self,
      problem_description: str,
      root_individual: evolution_database.Program,
      previous_individuals: Sequence[evolution_database.Program],
  ) -> str:
    """Samples a prompt by adding individuals from the database."""
    # Render the root individual and (unique) previous individuals.
    code = root_individual.render()
    previous_programs = set()
    for individual in previous_individuals:
      new_code = individual.render()
      if new_code not in previous_programs and new_code != code:
        previous_programs.add(new_code)

    # Sample the stochastic format arguments from the distribution.
    stochastic_args = json.loads(PROMPT_FORMAT_ARGS)
    format_kwargs = {}
    for (
        placeholder_name,
        placeholder_random_variables,
    ) in stochastic_args.items():
      values, probabilities = zip(*placeholder_random_variables)
      # Sample a format argument value from the distribution and replace it in
      # the prompt.
      id_ = random.choices(range(len(values)), weights=probabilities, k=1)[0]
      format_kwargs[placeholder_name] = values[int(id_)]

    # Format the prompt with previous programs, format args, and diff params.
    return PROMPT_TEMPLATE.format(
        code=code,
        previous_programs='\n'.join(previous_programs),
        problem_description=problem_description,
        **format_kwargs,
        **_DIFF_PARAMS,
    )
