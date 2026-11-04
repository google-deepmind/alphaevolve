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

"""Client for executing evolved code."""

import abc
import asyncio
from collections.abc import Mapping
import hashlib
import io
import json
import os
import tarfile
import tempfile
import textwrap
from typing import Any

from components import exceptions
import docker
from typing_extensions import override


class BaseEvaluator(abc.ABC):
  """Code evaluation interface."""

  def __init__(self, timeout_seconds: int = 2000):
    self._timeout_seconds = timeout_seconds

  def run(self, code: str, parent_output: Any | None = None) -> Any:
    """Runs the code and returns the response."""
    try:
      result = self._run(code, parent_output)
    except exceptions.CodeExecutionError:
      raise
    except Exception as e:  # pylint: disable=broad-exception-caught
      raise exceptions.CodeExecutionError(f'Code execution failed: {e}') from e
    if not isinstance(result, Mapping) or 'scores_to_maximize' not in result:
      raise exceptions.CodeExecutionError(
          f'Invalid result from evaluator: {result}.'
      )
    return result['scores_to_maximize'], result.get('output_artifacts', None)

  @abc.abstractmethod
  def _run(self, code: str, parent_output: Any | None = None) -> Any:
    """Runs the code and returns the response."""

  async def run_async(self, code: str, parent_output: Any | None = None) -> Any:
    """Runs the code and returns the response asynchronously."""
    return await asyncio.to_thread(self.run, code, parent_output)

  def get_state(self) -> Mapping[str, Any]:
    return {}

  def set_state(self, state: Mapping[str, Any]) -> None:
    del state  # Unused.


class DockerEvaluator(BaseEvaluator):
  """Evaluates code securely in an ephemeral Docker container."""

  _RESULT_SEPARATOR = '~~~~~~~RESULT_SEPARATOR~~~~~~~'

  def __init__(
      self,
      timeout_seconds: int = 2000,
      base_image: str = 'python:3.11',
      mem_limit: str = '512m',
      requirements_path: str | None = 'requirements.txt',
  ):
    super().__init__(timeout_seconds)
    self._base_image = base_image
    self._mem_limit = mem_limit
    self._requirements_path = requirements_path
    self._image = self._build_image_if_needed()

  def _build_image_if_needed(self) -> str:
    """Builds a custom image with requirements, or returns base image."""
    if not self._requirements_path:
      return self._base_image

    with open(self._requirements_path, 'r') as f:
      requirements_content = f.read()

    # Create a unique tag based on requirements hash
    req_hash = hashlib.md5(requirements_content.encode()).hexdigest()[:12]
    custom_image = f'alphaevolve-sandbox:{req_hash}'

    client = docker.from_env()

    # Check if image already exists
    try:
      client.images.get(custom_image)
      return custom_image
    except docker.errors.ImageNotFound:
      pass

    # Build custom image with requirements
    dockerfile = textwrap.dedent(f"""
        FROM {self._base_image}

        # Install tools
        RUN apt-get update && apt-get install -y \\
            gfortran \\
            build-essential \\
            && rm -rf /var/lib/apt/lists/*

        RUN useradd --create-home --shell /bin/bash sandbox

        COPY requirements.txt /tmp/requirements.txt
        RUN pip install --no-cache-dir -r /tmp/requirements.txt

        WORKDIR /home/sandbox

        USER sandbox
    """)
    # Create build context with Dockerfile and requirements.txt
    context = io.BytesIO()
    with tarfile.open(fileobj=context, mode='w') as tar:
      # Add Dockerfile
      dockerfile_bytes = dockerfile.encode('utf-8')
      dockerfile_info = tarfile.TarInfo(name='Dockerfile')
      dockerfile_info.size = len(dockerfile_bytes)
      tar.addfile(dockerfile_info, io.BytesIO(dockerfile_bytes))

      # Add requirements.txt
      requirements_bytes = requirements_content.encode('utf-8')
      requirements_info = tarfile.TarInfo(name='requirements.txt')
      requirements_info.size = len(requirements_bytes)
      tar.addfile(requirements_info, io.BytesIO(requirements_bytes))

    # Set position back to start of file for building image.
    context.seek(0)
    try:
      client.images.build(
          fileobj=context, custom_context=True, tag=custom_image
      )
    except docker.errors.BuildError as e:
      print('!!! DOCKER BUILD FAILED !!!')
      print('Here are the logs from inside the container build process:')
      for log in e.build_log:
        if 'stream' in log:
          print(log['stream'].strip())
      raise e

    return custom_image

  @override
  def _run(self, code: str, parent_output: Any | None = None) -> Any:
    """Runs code in a sandboxed Docker container."""
    script_content = self._build_script(code, parent_output)
    client = docker.from_env()

    with tempfile.TemporaryDirectory() as temp_dir:
      script_path = os.path.join(temp_dir, 'script.py')

      with open(script_path, 'w') as f:
        f.write(script_content)

      os.chmod(script_path, 0o644)

      container = None
      try:
        container = client.containers.run(
            image=self._image,
            command=[
                'timeout',
                '--signal=KILL',
                str(self._timeout_seconds + 60),
                'python',
                '/code/script.py',
            ],
            volumes={script_path: {'bind': '/code/script.py', 'mode': 'ro'}},
            log_config=docker.types.LogConfig(
                type='json-file', config={'max-size': '10m', 'max-file': '1'}
            ),
            network_disabled=True,
            read_only=True,
            cap_drop=['ALL'],
            security_opt=['no-new-privileges'],
            tmpfs={'/tmp': 'size=100m,mode=1777'},
            mem_limit=self._mem_limit,
            pids_limit=100,
            nano_cpus=1000000000,
            detach=True,
            user='sandbox',
            environment={
                'OPENBLAS_NUM_THREADS': '1',
                'OMP_NUM_THREADS': '1',
                'MKL_NUM_THREADS': '1',
                'VECLIB_MAXIMUM_THREADS': '1',
                'NUMEXPR_NUM_THREADS': '1',
            },
        )

        result = container.wait(timeout=self._timeout_seconds)

        logs_generator = container.logs(stdout=True, stderr=True, stream=True)
        output_accumulated = []
        total_bytes = 0
        max_log_size = 100 * 1024 * 1024  # 100 MB Limit

        try:
          for chunk in logs_generator:
            output_accumulated.append(chunk)
            total_bytes += len(chunk)
            if total_bytes > max_log_size:
              output_accumulated.append(b'\n... [TRUNCATED DUE TO SIZE] ...')
              break
        except Exception:  # pylint: disable=broad-exception-caught
          pass  # Handle stream errors gracefully

        full_log = b''.join(output_accumulated).decode(
            'utf-8', errors='replace'
        )

        if result['StatusCode'] != 0:
          raise RuntimeError(
              f'Container exited with code {result["StatusCode"]}: {full_log}'
          )

        return self._parse_result(full_log)
      finally:
        if container:
          container.remove(force=True)

  def _build_script(self, code: str, parent_output: Any | None) -> str:
    """Builds the script to execute inside the container."""

    try:
      parent_output_json = json.dumps(parent_output)
    except (TypeError, ValueError):
      parent_output_json = 'null'

    return textwrap.dedent("""
        import json

        try:
            raw_input = {parent_output_json}
            PARENT_OUTPUT = json.loads(raw_input)
        except Exception:
            PARENT_OUTPUT = None

        {code}

        if 'evaluate' not in dir():
            raise ValueError('Function "evaluate" not found.')

        result = evaluate()
        print("{result_separator}")
        print(json.dumps(result))
    """).format(
        code=code,
        result_separator=self._RESULT_SEPARATOR,
        parent_output_json=repr(parent_output_json),
    )

  def _parse_result(self, output: str) -> Any:
    """Parses the result from container stdout."""
    if self._RESULT_SEPARATOR not in output:
      raise RuntimeError(f'Result separator not found in output: {output}')

    result_json = output.split(self._RESULT_SEPARATOR)[-1].strip()
    return json.loads(result_json)
