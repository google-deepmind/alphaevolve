# AlphaEvolve

This repository accompanies the paper

TODO(franrruiz): Add link to paper.

There are the following directories:

- `/src` contains a minimal implementation of AlphaEvolve with local evaluation.

- `/omega` contains code and data related to the new upper bound on the matrix
multiplication exponent discovered by AlphaEvolve. It includes a Colab notebook
for rigorous verification using rational arithmetic, and a `level_3/`
subdirectory with the gradient-based optimisation code at recursion depth 3:

    - `level_3/final_program.py` — the optimisation program evolved by
    AlphaEvolve. It builds a differentiable upper-bound model in JAX/Haiku,
    runs gradient descent over 32k steps with 32 random seeds, and converts the
    best solution into a format suitable for verification. Run it directly with
    `python3 final_program.py`.

    - `level_3/verification.py` — a NumPy/SciPy verifier that takes a flat
    parameter vector produced by the optimiser and evaluates the proven omega
    upper bound. It uses floating-point arithmetic for fast hill-climbing; the
    companion Colab notebook uses rational arithmetic for a rigorous proof.

    A `requirements.txt` listing the pinned Python dependencies and a
    `Dockerfile` for containerised execution are also provided in `level_3/`.

- `/maths` contains the initial programs, discovered results, and verification
code for the mathematical problems on which AlphaEvolve found a solution
outperforming the state-of-the-art (see the "Results" section of the paper).
There is a subfolder for each of such problems, each containing the files needed
to run an AlphaEvolve experiment and a Colab notebook to verify the found
construction.

- `/matmul_tensor_decompositions` contains Colab notebooks with the
decompositions of matrix multiplication tensors across the matrix sizes for
which the gradient-based optimization method from AlphaEvolve found a solution
outperforming the state-of-the-art, along with the code to verify their
correctness. Specifically, there are two Colabs: one for the standard tensor
rank decomposition, and another one for the support rank.

## Installation

For the folders `/maths` and `/matmul_tensor_decompositions`, no
installation is required. The provided notebooks can be opened and run in Google
Colab.

For the folder `/src`, these are the installation instructions:

1. Install docker (required for program evaluation): https://www.docker.com/get-started/

2. (optional) Create venv:

```bash
sudo apt-get install virtualenv python3-venv
virtualenv -p python3.11 alpha_evolve
source alpha_evolve/bin/activate
```

3. Install pip dependencies:

```bash
pip install -r src/requirements.txt
```

4. Set [Gemini API key](https://ai.google.dev/gemini-api/docs/api-key):

```bash
export GEMINI_API_KEY=<your_api_key>
```

## Usage

- `/omega`: The Colab notebook to verify the new upper bound on the matrix
  multiplication exponent can be opened via:
    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/alphaevolve/blob/master/omega/verification.ipynb).
    _Note: To verify our best omega upper bound, you must download the
    `clean_level4.npz.xz` file from this repository to your local machine,
    and then upload it when prompted by the Colab notebook._

- `/maths`: The Colab notebook to verify the found solution for each problem can be opened via:

    - Circle packing:
    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/alphaevolve/blob/master/maths/circle_packing/verification.ipynb).

    - Erdös minimum overlap:
    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/alphaevolve/blob/master/maths/erdos_minimum_overlap/verification.ipynb).

    - First autocorrelation inequality:
    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/alphaevolve/blob/master/maths/first_autocorrelation_inequality/verification.ipynb).

    - Second autocorrelation inequality:
    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/alphaevolve/blob/master/maths/second_autocorrelation_inequality/verification.ipynb).

    - Third autocorrelation inequality:
    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/alphaevolve/blob/master/maths/third_autocorrelation_inequality/verification.ipynb).

    - Heilbronn problem for convex regions:
    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/alphaevolve/blob/master/maths/heilbronn_convex/verification.ipynb).

    - Heilbronn problem for triangles:
    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/alphaevolve/blob/master/maths/heilbronn_triangle/verification.ipynb).

    - Hexagon packing:
    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/alphaevolve/blob/master/maths/hexagon_packing/verification.ipynb).

    - Kissing numbers:
    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/alphaevolve/blob/master/maths/kissing/verification.ipynb).

    - Max to min ratio:
    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/alphaevolve/blob/master/maths/max_min_ratio/verification.ipynb).

    - Sums and differences:
    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/alphaevolve/blob/master/maths/sums_and_differences/verification.ipynb).

    - Uncertainty principles (Hermite polynomials approach):
    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/alphaevolve/blob/master/maths/uncertainty_principles/verification_hermite.ipynb)

    - Uncertainty principles (Laguerre polynomials approach):
    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/alphaevolve/blob/master/maths/uncertainty_principles/verification_laguerre.ipynb)

    - Extremal girth 5:
    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/alphaevolve/blob/master/maths/extremal_girth_5/verification.ipynb)

    - Shannon capacity of cycle graphs:
    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/alphaevolve/blob/master/maths/shannon_cycle/verification.ipynb)

    - Cap sets:
    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/alphaevolve/blob/master/maths/cap_sets/verification.ipynb)

- `/matmul_tensor_decompositions`: There are two Colab notebooks:

    - Standard rank:
    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/alphaevolve/blob/master/matmul_tensor_decompositions/verification_rank.ipynb).

    - Support rank:
    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/alphaevolve/blob/master/matmul_tensor_decompositions/verification_support_rank.ipynb).

- `/src`: We provide an example launch command (for the circle packing problem) below:

```bash
python run.py --problem_config=./data/problem_config/circle_packing.yaml
```

**DISCLAIMER:** AlphaEvolve involves the execution of code generated by Large
Language Models (LLMs). LLM-generated code is intrinsically unpredictable and
should be handled as untrusted code. In this version of AlphaEvolve, this code
is run in a Docker sandbox as an isolation measure. However, despite sandboxing,
the potential for unintended behavior, resource misuse, or security
vulnerabilities cannot be fully eliminated. All users should be aware of these
inherent risks and understand that they use this software at their own
discretion. Google DeepMind shall not be liable for any claims, damages, or
other liability arising from the execution of LLM-generated code.

## Citing this work

If you use the code or data in this package, please cite:

```
TODO(franrruiz): Add citation.
```

## License and disclaimer

Copyright 2026 Google LLC

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
