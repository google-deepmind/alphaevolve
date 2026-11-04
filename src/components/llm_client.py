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

"""LLM proxy for self-hosted models."""

import abc
import asyncio
from collections.abc import Mapping, Sequence
import concurrent.futures
import os
import random
from typing import Any

from components import exceptions
from google import genai
from google.genai import types as genai_types
from typing_extensions import override


class BaseLLMClient(abc.ABC):
  """Base class for LLM clients."""

  @abc.abstractmethod
  def generate(
      self,
      prompt: str,
      model_name: str | None = None,
      system_instruction: str | None = None,
  ) -> str:
    """Generates a response from the LLM."""

  async def generate_async(
      self,
      prompt: str,
      model_name: str | None = None,
      system_instruction: str | None = None,
  ) -> str:
    """Generates a response from the LLM asynchronously."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
      future = asyncio.get_running_loop().run_in_executor(
          executor,
          self.generate,
          prompt,
          model_name,
          system_instruction,
      )
      return await future

  def get_state(self) -> Mapping[str, Any]:
    return {}

  def set_state(self, state: Mapping[str, Any]) -> None:
    del state  # Unused.


class GenAILLMClient(BaseLLMClient):
  """LLM generator using google.genai with Gemini."""

  def __init__(
      self,
      api_key: str | None = None,
      # NOTE: These are the models that were used at the time of release. Some
      # of them may since have been deprecated or shut down; please update this
      # list to models that are currently available in the Gemini API.
      model_weights: Sequence[tuple[str, float]] = (
          ('gemini-2.5-flash', 0.7),
          ('gemini-2.5-pro', 0.2),
          ('gemini-3-pro-preview', 0.1),
      ),
  ):
    if api_key is None:
      api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
      raise ValueError(
          'GEMINI_API_KEY environment variable not set and no api_key provided.'
      )
    self._client = genai.Client(api_key=api_key)
    self._model_weights = model_weights

  @override
  def generate(
      self,
      prompt: str,
      model_name: str | None = None,
      system_instruction: str | None = None,
  ) -> str:
    """Generates a response from the LLM."""
    if model_name is None:
      # Sample a model name based on weights.
      models, weights = zip(*self._model_weights)
      model_name = random.choices(models, weights=weights, k=1)[0]

    config = None
    if system_instruction is not None:
      config = genai_types.GenerateContentConfig(
          system_instruction=system_instruction,
      )

    try:
      response = self._client.models.generate_content(
          model=model_name,
          contents=prompt,
          config=config,
      )
      return response.text
    except exceptions.LLMInferenceError:
      raise
    except (genai.errors.APIError, ValueError) as e:
      raise exceptions.LLMInferenceError(
          f'LLM inference failed for model {model_name}: {e}'
      ) from e
