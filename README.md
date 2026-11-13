# Topic-Modeling-Theories

This repository provides an in-depth exploration of various topic modeling techniques, implemented using Python and Jupyter Notebooks(or Jupyter lab). The following models are covered:

- **Latent Dirichlet Allocation (LDA)**
- **Dynamic Topic Model (DTM)**
- **Topics Over Time (ToT)**
- **Correlated Topic Model (CTM)**
- **Structural Topic Model (STM)**
- **Biterm Topic Model (BTM)**
- **BERTopic**

## Getting Started

### Prerequisites

- **Make**: Before setting up the environment, ensure that `Make` is installed on your system.
  For **Windows users**, please refer to [this guide](https://gnuwin32.sourceforge.net/packages/make.htm) to install `Make`.

- **Python 3** with `pip`, on your `PATH`.

- **R with `Rscript`**, only if you plan to run `notebook/Structural_Topic_Model.ipynb` or `notebook/Topic_over_Times.ipynb` (the notebooks implemented in R, via the `stm` and `topicovertime` packages). If you don't need them, skip `make install-r` below.

- **Mecab (Korean morphological analyzer)**, only if you plan to run `notebook/BERTopic.ipynb`. The `mecab-python3` package installed by `requirements.txt` is a Python binding; it still requires the Mecab binary and a Korean dictionary to be installed at the OS level (via `konlpy`'s [Mecab installation guide](https://konlpy.org/en/latest/install/)). This step is notoriously more involved on Windows than on Linux/macOS.

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/sorrychoe/topic-modeling-theory.git
   cd topic-modeling-theory
   ```

2. Initialize the setup:
   ```bash
   make init
   ```

3. Install dependencies:
   ```bash
   make install          # everything, including the R setup for Structural_Topic_Model.ipynb & Topic_over_Times.ipynb
   # or, if you don't have R and only want the Python notebooks:
   make install-python
   ```

Now you're ready to start exploring the topic modeling notebooks.

> `requirements.txt` pins lower bounds (`>=`) rather than exact versions, so `make install` can resolve wheels that actually exist for whatever Python version you're on. One consequence: if pip picks newer library versions than the ones the notebooks were originally run with (numpy 1.26.4, pandas 2.0.0, tomotopy 0.12.7, bertopic 0.16.4, etc.), the exact topic wording/counts you get from re-running a notebook may differ slightly from the cached output already in it, even with the seed fixed.

## Available Notebooks

Each model is implemented in a separate Jupyter Notebook, providing explanations and code walkthroughs. You can start by running any of the notebooks to see how the models work and modify the code to fit your specific use case.

1. **notebook/Latent_Dirichlet_Allocation.ipynb**:
   - Theory & Python Sample Code of Latent Dirichlet Allocation.
2. **notebook/Dynamic_Topic_Model.ipynb**:
   - Theory & Python Sample Code of Dynamic Topic Model.
3. **notebook/Topic_over_Times.ipynb**:
   - Theory & R Sample Code of Topics Over Time.
4. **notebook/Correlated_Topic_Model.ipynb**:
   - Theory & Python Sample Code of Correlated Topic Model.
5. **notebook/Structural_Topic_Model.ipynb**:
    - Theory & R Sample Code of Structural Topic Model.
6. **notebook/Biterm_Topic_Model.ipynb**:
    - Theory & Python Sample Code of Biterm Topic Model.
7. **notebook/BERTopic.ipynb**:
   - Theory & Python Sample Code of BERTopic.

Each notebook's code cells are also mirrored as a plain script under `notebook/src/` (e.g. `notebook/src/BERTopic.py`, or `notebook/src/Topic_over_Times.R` for the R notebooks) for readers who want to run or diff the pipeline outside Jupyter.

## Authors

- [@sorrychoe](https://www.github.com/sorrychoe)

## you have some issue?

If there is a problem while using it, please register the issue section in Github.
