## Usage

### Installation

1. Pre-install step:
   ```
   pip install conda-merge
   ```
2. Create conda environment:
   1. GPU machines:
      ```
      conda-merge env.common.yaml env.gpu.yaml > env.yaml
      conda env create -f env.yaml
      conda activate matdeeplearn
      ```
   
   2. CPU-only machines:

      1. M1 Macs (see https://github.com/pyg-team/pytorch_geometric/issues/4549):
         ```
         conda env create -f env.common.yaml
         conda activate matdeeplearn
         MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ pip install torch-scatter torch-sparse torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cpu.html
         ```
         Note: if pip is using cached wheels and the build is failing, add `--no-cache-dir` flag.

      2. Everyone else:
         ```
         conda-merge env.common.yaml env.cpu.yaml > env.yaml
         conda env create -f env.yaml
         conda activate matdeeplearn
         ```

3. Install package:
   ```
   pip install -e .
   ```

## Development

#### Code Quality
This project uses flake8, black, and isort for linting.
To install the pre-commit git hook, run:
```
pre-commit install
```
By default, the hooks will run every time you say:
```
git commit -m "Commit message"
```
However, for more information, please see: https://pre-commit.com/#usage
