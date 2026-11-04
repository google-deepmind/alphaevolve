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

"""Parsing utils for AlphaEvolve."""

from collections.abc import Sequence
import dataclasses
import functools
import re

from components import exceptions


SEARCH = '<<<<<<< SEARCH'
DIVIDER = '======='
REPLACE = '>>>>>>> REPLACE'
_EVOLVE_BLOCK_START = '# EVOLVE-BLOCK-START'
_EVOLVE_BLOCK_END = '# EVOLVE-BLOCK-END'
_CODE_BLOCK_START = '```python'
_CODE_BLOCK_END = '```'


def _block_re_pattern(content: str, del_l: str, del_r: str) -> str:
  return f'{del_l}\n{content}{del_r}'


evolve_block_pattern = functools.partial(
    _block_re_pattern, del_l=_EVOLVE_BLOCK_START, del_r=_EVOLVE_BLOCK_END
)
code_block_pattern = functools.partial(
    _block_re_pattern, del_l=_CODE_BLOCK_START, del_r=_CODE_BLOCK_END
)


@dataclasses.dataclass(frozen=False, kw_only=True)
class EvolveBlock:
  """A block of several lines of code to be evolved.

  This is used to parse the code into blocks, indicated by the delimiters
  `EVOLVE-BLOCK-START` and `EVOLVE-BLOCK-END`. For each block, we extract the
  code between the delimiters as is_mutable=True and the remaining code as
  is_mutable=False.

  Attributes:
    code: The code in the block.
    is_mutable: Whether the code is mutable, i.e. editable by the LLM.
  """

  code: str
  is_mutable: bool


def _parse_evolve_blocks(
    code: str,
    block_start: str,
    block_end: str,
    evolve_entire_file: bool,
) -> Sequence[EvolveBlock]:
  r"""Parses the code into blocks to be evolved.

  Parse the code into blocks, indicated by the delimiters `block_start` and
  `block_end`. For each block, we extract the code between the delimiters as
  is_mutable=True and the remaining code as is_mutable=False. For each
  matching delimiter, we discard the rest of the line.

  For example, the code:

  ```python
  a = 1
  # EVOLVE-BLOCK-START
  b = 2
  # EVOLVE-BLOCK-END
  c = 3
  # EVOLVE-BLOCK-START
  d = 4
  e = 5
  # EVOLVE-BLOCK-END
  ```

  will be parsed into:
  [
      EvolveBlock(code='a = 1\n# EVOLVE-BLOCK-START\n', is_mutable=False),
      EvolveBlock(code='b = 2\n', is_mutable=True),
      EvolveBlock(code='# EVOLVE-BLOCK-END\nc = 3\n# EVOLVE-BLOCK-START\n', \
        is_mutable=False),
      EvolveBlock(code='d = 4\ne = 5\n', is_mutable=True),
      EvolveBlock(code='# EVOLVE-BLOCK-END', is_mutable=False)
  ]

  Note that the code between the delimiters will be modified by the LLM,
  while the remaining code will be hidden.

  Args:
    code: The code to parse.
    block_start: The delimiter to indicate the start of a block.
    block_end: The delimiter to indicate the end of a block.
    evolve_entire_file: Whether to evolve the entire file or just the blocks.

  Returns:
    A sequence of blocks to be evolved. Mutable blocks are indicated by
    is_mutable=True, while the remaining code is indicated by
    is_mutable=False.
  """
  if evolve_entire_file:
    if block_start in code or block_end in code:
      raise ValueError(
          'Evolving the full file is not supported when block delimiters are'
          ' present in the code.'
      )
    return [EvolveBlock(code=code, is_mutable=True)]
  if block_start not in code and block_end not in code:
    raise ValueError(
        'No evolve block delimiters found in the code. If you want to'
        ' evolve the full file, set `evolve_entire_file=True`.'
    )
  blocks: list[EvolveBlock] = []
  lines = code.splitlines(keepends=True)
  in_mutable_block = False
  current_block_code = ''

  for i_line, line in enumerate(lines):
    has_start = block_start in line
    has_end = block_end in line
    if has_start and has_end:
      raise ValueError(
          f'Parsing error: Unexpected delimiters in line {i_line}: {line}'
      )
    elif has_start:
      if in_mutable_block:
        raise ValueError(
            f'Parsing error: Unexpected delimiter in line {i_line}: {line}'
        )
      current_block_code += line
      in_mutable_block = True
      if current_block_code:
        blocks.append(EvolveBlock(code=current_block_code, is_mutable=False))
        current_block_code = ''
    elif has_end:
      if not in_mutable_block:
        raise ValueError(
            f'Parsing error: Unexpected delimiter in line {i_line}: {line}'
        )
      in_mutable_block = False
      if current_block_code:
        blocks.append(EvolveBlock(code=current_block_code, is_mutable=True))
        current_block_code = ''
      current_block_code += line
    else:  # No start or end delimiter.
      current_block_code += line

  if in_mutable_block:
    raise ValueError('Parsing error: Unexpected end of file while in a block.')
  if current_block_code:
    blocks.append(EvolveBlock(code=current_block_code, is_mutable=False))
  return blocks


def _find_edit_blocks(block: str) -> Sequence[tuple[str, str]]:
  """Splits a block of edits into individual edits.

  Example:
  block:
  <<<<<<< SEARCH
  This is the original text.
  =======
  This is the updated text.
  >>>>>>> REPLACE

  Output:
  [('This is the original text.', 'This is the updated text.')]

  Args:
    block: The block of edits.

  Returns:
    A list of (original, updated) pairs.
  """
  # Check that the block is properly formatted with equal non-zero numbers of
  # separators:
  search_count = block.count(SEARCH)
  divider_count = block.count(DIVIDER)
  replace_count = block.count(REPLACE)
  if not (search_count == divider_count == replace_count > 0):
    return []

  remaining_text = block
  # Separator regexen, including trailing whitespace:
  search_separator = SEARCH + ' *\n'
  divider_separator = DIVIDER + ' *\n'
  replace_separator = REPLACE + ' *\n?'

  old_and_new_text_pairs = []
  while remaining_text:
    # This loops through the block, splitting off the section from the next
    # delimiter, recording it (if desired) and then moving on to the next
    # delimiter.
    common_previous_text, *search_and_remainders = re.split(
        search_separator, remaining_text, maxsplit=1)
    del common_previous_text  # Unused text from before the <<SEARCH.

    if not search_and_remainders:
      # Nothing more to parse.
      break
    else:
      search_and_remainder = search_and_remainders.pop(0)
      # This assert holds because maxsplit:=1 and the length-0 case hits the
      # above break invoking case.
      assert not search_and_remainders, search_and_remainders

    old_text, remaining_text = re.split(
        divider_separator, search_and_remainder, maxsplit=1)

    new_text, remaining_text = re.split(
        replace_separator, remaining_text, maxsplit=1)

    old_and_new_text_pairs.append((old_text, new_text))

  return old_and_new_text_pairs


def _apply_edits(original_code: str, edits: Sequence[tuple[str, str]]) -> str:
  """Applies edits to a piece of code."""
  new_code = original_code
  for before, after in edits:
    new_code = new_code.replace(before, after, 1)
  return new_code


def apply_diffs(
    code: str,
    llm_sample: str,
) -> str:
  """Mutates the program by applying the diffs in the LLM sample."""
  # Decompose the code into `EvolveBlock` blocks.
  evolve_code_blocks = _parse_evolve_blocks(
      code,
      block_start='# EVOLVE-BLOCK-START',
      block_end='# EVOLVE-BLOCK-END',
      evolve_entire_file=False,
  )

  # Extract code blocks from the LLM sample.
  diff_instruction_blocks = re.findall(
      code_block_pattern('(.*?)'), llm_sample, re.DOTALL
  )
  diff_edits = []
  try:
    for diff_block in diff_instruction_blocks:
      diff_edits += _find_edit_blocks(diff_block)
  except exceptions.DiffParsingError:
    raise
  except (ValueError, AssertionError) as e:
    raise exceptions.DiffParsingError(
        f'Failed to parse diffs from LLM output: {e}'
    ) from e

  # Apply the edits to the code.
  for evolve_block in evolve_code_blocks:
    if evolve_block.is_mutable:
      evolve_block.code = _apply_edits(
          evolve_block.code, diff_edits
      )
  return ''.join([block.code for block in evolve_code_blocks])
