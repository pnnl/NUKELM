# NukeLM

Utilities for training, serving, and inspecting the NukeLM models.
For access to the model weights, please email nukelm@pnnl.gov and provide your name and institution.


## Set-Up
We recommend using `conda` to create a virtual environment, then installing dependencies with `pip`.
Only Python 3.8 is tested, but other versions may work without modification.
Basic dependencies are listed in `requirements.txt`,
and additional development dependencies are listed in `requirements-dev.txt`.
The following commands will setup a virtual environment and install all dependencies from both lists.

```bash
conda create -n nukelm "python<3.9" pip
conda activate nukelm
pip install -e .[dev]
```

To use the NukeLM training notebooks, download example training scripts from
[Huggingface transformers](https://github.com/huggingface/transformers/tree/main/examples/pytorch)
and place them as below.
Use the version of the training script that matches your distribution of `transformers`.
This repository was tested against transformers==4.8.1.

* `examples/pytorch/text-classification/run_glue.py` --> `src/nukelm/finetune/run_glue.py`
* `examples/pytorch/language-modeling/run_mlm.py` --> `src/nukelm/pretrain/run_mlm.py`

### GPU Support
The NukeLM training and serving code supports GPU processing, but dependency installation may be slightly different.
The easiest solution may be to install `pytorch` from the "pytorch" channel with `Conda`.
Find your cuda version with, e.g., `nvidia-smi` (here shown as '10.1'),
then replace the above command with
```bash
conda create -n nukelm "python<3.9" pip pytorch cudatoolkit=10.1 -c pytorch
conda activate nukelm
pip install -e .[dev]
```

## Models
Example models are distributed separately from this repository.
Place them in `data/06_models` for interoperability with the scripts and notebooks.

## Usage

### 1. Training

#### 1a. Command Line
To pre-train a model, use `nukelm-pretrain [ARGS]`. To fine-tune a model, use 
`nukelm-finetune [ARGS]`. In either case, pass the argument "--help" to see available options.

#### 1b. In a Python interpreter
The scripts can be imported via `nukelm.run_fine_tune` and `nukelm.run_language_modeling`, both of which 
accept a path to a JSON-formatted configuration file with the same options as the CLI.

#### 1c. Databricks
For use in Azure Databricks, see the example notebooks
* `notebooks/Databricks - Pre-Train.ipynb`
* `notebooks/Databricks - Fine-Tune.ipynb`
* `notebooks/Databricks - Hyperparameter Search.ipynb`
* `notebooks/Databricks - Fine-Tune Scaling Experiment.ipynb`

### Warnings when loading saved models
Huggingface's transformers package, starting with v3.0.0, displays a warning
if not all saved weights are used, or not all expected weights are initialized.
The former may occur with NukeLM fine-tuned models, because the pooling layer in
the underlying `RobertaModel` is not used for Sequence Classification tasks.
The warning may be safely ignored.
[Reference GitHub issue](https://github.com/huggingface/transformers/issues/5421#issuecomment-715383840).

### 2. Model Serving
For examples of performing masked language modeling (MLM) or document classification, see the example notebooks
* `notebooks/MLM Examples`
* `notebooks/Classification Examples`

To apply models to a whole set of documents, 
first, save the text in a CSV compatible with Huggingface's `datasets`, with the text in a column labeled "text".
Then, use `nukelm-serve [ARGS]`. Pass the argument "--help" to see available options;
at minimum, these include the path to the CSV, a model name or path, and where to store the output.
Be default, it outputs document embeddings aggregated across tokens in three ways,
the predicted probability distribution across all labels, and the label predicted most likely.

For example, if the NukeLM binary classifier is saved to `data/06_models/Binary Classification` and example data
is saved to `data/03_primary/example-osti-abstracts.csv`, the following command will serve the model on those data,
saving the output to `data/07_model_output/binary-classification`.
```bash
nukelm-serve --model-path "data/06_models/Binary Classification" -o binary-classification -i data/03_primary/example-osti-abstracts.csv --tokenizer-name roberta-large
```
The output is saved with Huggingface's `datasets` and can be loaded with, e.g., `datasets.load_from_disk("data/07_model_output/binary-classification")`.

These embeddings and labels can be used in further analysis, like BERTopic.

### 3. Analysis
For an example applying BERTopic, see the example notebook `notebooks/BERTopic Exploration - OSTI.ipynb`.

## Contributing
1. Set up [pre-commit](https://pre-commit.com/) hooks to run before any commit
   ```
   pre-commit install
   pre-commit autoupdate
   ```
   These are defined in `.pre-commit-config.yaml` and configured in `setup.cfg` and `pyproject.toml`.
   Run `pre-commit run --all-files` to run the checks against the whole repository, at any time.
2. Build documentation
   ```
   python -c "from nukelm.utils import build_docs; build_docs()"
   ```
3. Run tests and generate a coverage report.
   ```
   pytest -v --cov-report html --cov-report term --cov src/nukelm --cov-config setup.cfg --junitxml coverage.xml src/tests
   ```

## Data organization
This project borrows its organization of the `data` folder from [Kedro](https://github.com/quantumblacklabs/kedro). 
The following table documents its usage. Note that by "data model" this project will mean "serialized data" -- Kedro 
allows for other methods of storing data like cloud resources and Spark tables.

```eval_rst
+----------------+---------------------------------------------------------------------------------------------------+
| Folder in data | Description                                                                                       |
+================+===================================================================================================+
| Raw            | Initial start of the pipeline, containing the sourced data model(s) that should never be changed, |
|                | it forms your single source of truth to work from. These data models are typically un-typed in    |
|                | most cases e.g. csv, but this will vary from case to case.                                        |
+----------------+---------------------------------------------------------------------------------------------------+
| Intermediate   | Optional data model(s), which are introduced to type your :code:`raw` data model(s), e.g.         |
|                | converting string based values into their current typed representation.                           |
+----------------+---------------------------------------------------------------------------------------------------+
| Primary        | Domain specific data model(s) containing cleansed, transformed and wrangled data from either      |
|                | :code:`raw` or :code:`intermediate`, which forms your layer that you input into your feature      |
|                | engineering.                                                                                      |
+----------------+---------------------------------------------------------------------------------------------------+
| Feature        | Analytics specific data model(s) containing a set of features defined against the :code:`primary` |
|                | data, which are grouped by feature area of analysis and stored against a common dimension.        |
+----------------+---------------------------------------------------------------------------------------------------+
| Model input    | Analytics specific data model(s) containing all :code:`feature` data against a common dimension   |
|                | and in the case of live projects against an analytics run date to ensure that you track the       |
|                | historical changes of the features over time.                                                     |
+----------------+---------------------------------------------------------------------------------------------------+
| Models         | Stored, serialised pre-trained machine learning models.                                           |
+----------------+---------------------------------------------------------------------------------------------------+
| Model output   | Analytics specific data model(s) containing the results generated by the model based on the       |
|                | :code:`model input` data.                                                                         |
+----------------+---------------------------------------------------------------------------------------------------+
| Reporting      | Reporting data model(s) that are used to combine a set of :code:`primary`, :code:`feature`,       |
|                | :code:`model input` and :code:`model output` data used to drive the dashboard and the views       |
|                | constructed. It encapsulates and removes the need to define any blending or joining of data,      |
|                | improve performance and replacement of presentation layer without having to redefine the data     |
|                | models.                                                                                           |
+----------------+---------------------------------------------------------------------------------------------------+
```

## Project Organization
    ├── .gitignore               <- Configures Git tracking
    ├── .isort.cfg               <- Configures automatic sorting of import statements
    ├── .pre-commit-config.yaml  <- Configures Git pre-commit hooks
    ├── LICENSE                  <- Terms of use
    ├── README.md                <- The top-level README for developers using this project.
    ├── environment.yaml         <- Conda environment file for reproducing the analysis environment
    ├── setup.cfg                <- Configures formatting and linting tools
    ├── setup.py                 <- Makes project pip installable
    │
    ├── conf                     <- Central location for configuration files
    │
    ├── data                     <- Central location for all data
    │   ├── 01_raw
    │   ├── 02_intermediate
    │   ├── 03_primary
    │   ├── 04_features
    │   ├── 05_model_input
    │   ├── 06_models
    │   ├── 07_model_output
    │   └── 08_reporting
    ├── docs                     <- Documentation
    │   └── source               <- Source code for documentation
    │
    ├── logs                     <- Central location for all logging output
    │
    ├── notebooks                <- Jupyter notebooks for experimentation and UI
    │
    └── src                      <- Source code for use in this project
        ├── tests                <- Test suite
        └── nukelm               <- Package source code





## Disclaimer
This material was prepared as an account of work sponsored by an agency of the United States Government.  Neither the United States Government nor the United States Department of Energy, nor Battelle, nor any of their employees, nor any jurisdiction or organization that has cooperated in the development of these materials, makes any warranty, express or implied, or assumes any legal liability or responsibility for the accuracy, completeness, or usefulness or any information, apparatus, product, software, or process disclosed, or represents that its use would not infringe privately owned rights. Reference herein to any specific commercial product, process, or service by trade name, trademark, manufacturer, or otherwise does not necessarily constitute or imply its endorsement, recommendation, or favoring by the United States Government or any agency thereof, or Battelle Memorial Institute. The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States Government or any agency thereof.

PACIFIC NORTHWEST NATIONAL LABORATORY operated by BATTELLE for the UNITED STATES DEPARTMENT OF ENERGY under Contract DE-AC05-76RL01830.




## Copyright
Copyright 2023 Battelle Memorial Institute

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
