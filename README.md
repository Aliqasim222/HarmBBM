# HarmBBM

**HarmBBM: Harmonic Barzilai--Borwein Momentum for Deep Neural Network Training**

This repository contains the reproducible PyTorch implementation used to compare
HarmBBM with SGDM, Adam, AdaBelief, and BBbound on MNIST and PathMNIST. The code
is organized as a conventional public research repository, with separate model,
optimizer, data-loading, training, search, plotting, and testing modules.

## Authors

| Author | Affiliation |
|---|---|
| **Ali Raza** | School of Science, Hebei University of Technology, Tianjin, China | 
| **Xinwei Liu*** | School of Science, Hebei University of Technology, Tianjin, China | 
| **Dileep Kumar** | Department of Electronic Engineering, Faculty of Engineering, The Islamia University of Bahawalpur, Bahawalpur, Pakistan  |

`*` Corresponding author.

## Supported experiments

| Dataset | Model | Final seeds | Default epochs |
|---|---|---:|---:|
| MNIST | MLP-1024x1024 | 42, 123, 2026 | 150 |
| PathMNIST | ResNet18 adapted to 28x28 images | 42, 123, 2026 | 150 |

The five supported optimizers are:

```text
sgdm, adam, adabelief, bbbound, harmbbm
```

## Repository structure

```text
HarmBBM_GitHub/
├── .github/workflows/tests.yml
├── configs/
│   ├── mnist.yaml
│   └── pathmnist.yaml
├── harmbbm/
│   ├── data/
│   │   ├── common.py
│   │   ├── mnist.py
│   │   └── pathmnist.py
│   ├── engine/
│   │   ├── core.py
│   │   ├── results.py
│   │   └── runner.py
│   ├── models/
│   │   ├── mlp.py
│   │   └── resnet.py
│   ├── optimizers/
│   │   ├── adabelief.py
│   │   ├── bbbound.py
│   │   ├── factory.py
│   │   └── harmbbm.py
│   ├── search/
│   │   └── spaces.py
│   └── utils/
│       ├── io.py
│       ├── plotting.py
│       └── reproducibility.py
├── paper/README.md
├── scripts/
│   ├── plot_comparison.py
│   ├── run_experiment.py
│   ├── run_mnist.py
│   ├── run_pathmnist.py
│   └── smoke_test.py
├── tests/
├── CITATION.cff
├── LICENSE
├── README.md
├── environment.yml
├── pyproject.toml
└── requirements.txt
```

## Installation

### pip

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### conda

```bash
conda env create -f environment.yml
conda activate harmbbm
pip install -e .
```

Run the synthetic check before downloading any dataset:

```bash
python -m scripts.smoke_test
pytest
```

## Quick tests

MNIST, one seed, two epochs:

```bash
python -m scripts.run_mnist --quick --optimizers harmbbm,bbbound
```

PathMNIST, one seed, two epochs:

```bash
python -m scripts.run_pathmnist --quick --optimizers harmbbm,bbbound
```

The first PathMNIST run downloads the MedMNIST files unless
`--no-download` is supplied.

## Full experiments without hyperparameter search

Run all five methods on MNIST:

```bash
python -m scripts.run_mnist
```

Run all five methods on PathMNIST:

```bash
python -m scripts.run_pathmnist
```

Run only the proposed method:

```bash
python -m scripts.run_mnist --optimizers harmbbm
python -m scripts.run_pathmnist --optimizers harmbbm
```

Run only BBbound:

```bash
python -m scripts.run_mnist --optimizers bbbound
python -m scripts.run_pathmnist --optimizers bbbound
```

Custom output and data directories:

```bash
python -m scripts.run_mnist \
  --data-dir /kaggle/working/data \
  --output-dir /kaggle/working/harmbbm_results
```

## Validation-based hyperparameter search

Search is disabled in the default YAML files. The paper protocol evaluates 50
configurations for 30 epochs using seed 42:

```bash
python -m scripts.run_mnist \
  --do-search \
  --search-budget 50 \
  --search-epochs 30 \
  --tune-seed 42
```

For each optimizer--dataset pair, this gives

```text
50 configurations x 30 epochs = 1500 search epoch-equivalents
```

The selected configuration is then fixed and evaluated for 150 epochs using
seeds 42, 123, and 2026. Candidates are ranked by highest validation accuracy,
then lower validation loss, then earlier best-validation epoch. The test set is
not evaluated during search.

To search only HarmBBM:

```bash
python -m scripts.run_mnist \
  --optimizers harmbbm \
  --do-search \
  --search-budget 50 \
  --search-epochs 30
```

Search progress and the selected configuration are saved after every completed
trial in CSV, JSON, and Pickle formats.

## Kaggle example

```python
!unzip -q /kaggle/input/harmbbm-code/HarmBBM_GitHub_Repository.zip -d /kaggle/working/
%cd /kaggle/working/HarmBBM_GitHub
!pip install -q -r requirements.txt
!pip install -q -e .
!python -m scripts.smoke_test
```

MNIST:

```python
!python -m scripts.run_mnist \
  --optimizers harmbbm,bbbound \
  --epochs 150 \
  --seeds 42,123,2026 \
  --data-dir /kaggle/working/data \
  --output-dir /kaggle/working/results
```

PathMNIST:

```python
!python -m scripts.run_pathmnist \
  --optimizers harmbbm,bbbound \
  --epochs 150 \
  --seeds 42,123,2026 \
  --data-dir /kaggle/working/data \
  --output-dir /kaggle/working/results
```

## Creating comparison figures

After all five methods have completed:

```bash
python -m scripts.plot_comparison \
  --dataset mnist \
  --results-root ./results \
  --out-dir ./figures/mnist

python -m scripts.plot_comparison \
  --dataset pathmnist \
  --results-root ./results \
  --out-dir ./figures/pathmnist
```

The script generates:

- best-seed training-loss comparison, PNG at 600 DPI and PDF;
- best-seed validation-accuracy comparison, PNG at 600 DPI and PDF;
- three-seed mean test accuracy with sample-standard-deviation bars.

## References

1. J. Barzilai and J. M. Borwein, “Two-point step size gradient methods,”
   *IMA Journal of Numerical Analysis*, 1988.
2. I. Sutskever, J. Martens, G. Dahl, and G. Hinton, “On the importance of
   initialization and momentum in deep learning,” ICML, 2013.
3. D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” ICLR,
   2015.
4. J. Zhuang et al., “AdaBelief optimizer: Adapting stepsizes by the belief in
   observed gradients,” NeurIPS, 2020.
5. C. Tan, S. Ma, Y.-H. Dai, and Y. Qian, “Barzilai--Borwein step size for
   stochastic gradient descent,” NeurIPS, 2016.
6. Z.-J. Wang et al., “Adaptive learning rate optimization algorithms with
   dynamic bound based on Barzilai--Borwein method,” *Information Sciences*,
   vol. 634, pp. 42--54, 2023.

## License

This repository is released under the MIT License. See [`LICENSE`](LICENSE).
