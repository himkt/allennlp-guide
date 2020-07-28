---
title: 'Hyperparameter optimization with optuna'
description: "This chapter gives a basic tutorial for optimizing hyperparameters of your model."
type: chapter
---

<textblock>

In this chapter, we'll give a quick tutorial of hyperparameter optimization for your AllenNLP model.

</textblock>

<exercise id="1" title="Hyperparameter matters!">

A choice of hyperparameter sometimes has a strong impact for the performance of a model.


</exercise>

<exercise id="2" title="Building your model">

This tutorial work on sentiment analysis, one kind of text classification problem.
We use IMDb review dataset, which contains 20,000 reviews for training and 5,000 reviews for validating the performance of trained model.
If you haven't read <a href="https://guide.allennlp.org/your-first-model#1">the previous tutorial for text classification</a>, it would help you.

I show a sample configuration of a CNN-based classifier below.
This mdoel has 6 hyperparameters: `embedding_dim`, `dropout`, `lr`, `max_filter_size`, `num_filters`, and `output_dim`.

```json
local batch_size = 64;
local cuda_device = 0;
local num_epochs = 15;
local seed = 42;
local train_data_path = 'https://s3-us-west-2.amazonaws.com/allennlp/datasets/imdb/train.jsonl';
local validation_data_path = 'https://s3-us-west-2.amazonaws.com/allennlp/datasets/imdb/dev.jsonl';

local embedding_dim = 128;
local dropout = 0.2;
local lr = 0.1;
local max_filter_size = 4;
local num_filters = 128;
local output_dim = 128;
local ngram_filter_sizes = std.range(2, max_filter_size);

{
  numpy_seed: seed,
  pytorch_seed: seed,
  random_seed: seed,
  dataset_reader: {
    type: 'text_classification_json',
    tokenizer: {
      type: 'whitespace',
    },
    token_indexers: {
      tokens: {
        type: 'single_id',
      },
    },
  },
  datasets_for_vocab_creation: ['train'],
  train_data_path: train_data_path,
  validation_data_path: validation_data_path,
  model: {
    type: 'basic_classifier',
    text_field_embedder: {
      token_embedders: {
        tokens: {
          embedding_dim: embedding_dim,
        },
      },
    },
    seq2vec_encoder: {
      type: 'cnn',
      embedding_dim: embedding_dim,
      ngram_filter_sizes: ngram_filter_sizes,
      num_filters: num_filters,
      output_dim: output_dim,
    },
    dropout: dropout,
  },
  data_loader: {
    batch_size: batch_size,
  },
  trainer: {
    cuda_device: cuda_device,
    num_epochs: num_epochs,
    optimizer: {
      lr: lr,
      type: 'sgd',
    },
    validation_metric: '+accuracy',
  },
}
```

</exercise>

<exercise id="3" title="Optuna: hyperparameter optimization library">

Let's start to optimize hyperparameters of our model!

We introduce <a href="https://optuna.org">`Optuna`</a>, a hyperparameter optimization library for machine learning.
Optuna implements many search algorithm for parameters such as Tree-structured Parzen Estimator (TPE),
CMA Evolution Strategy (CMA-ES), and Multi-objective optimization.

Optuna also has a integration for AllenNLP, which is called <a href="https://optuna.readthedocs.io/en/stable/reference/integration.html#optuna.integration.AllenNLPExecutor">`AllenNLPExecutor`</a>.
We can use `AllenNLPExecutor` by following two steps: `Masking parameters` and `Defining search space`.

</exercise>

<exercise id="3" title="Masking parameters: preparing for Optuna">

First, we replace values of hyperparameters with `std.extVar` for tell Optuna what parameters to be optimized.
Remember that call `std.parseInt` or `std.parseFloat` for numerical parameters.

## Before

```json
local embedding_dim = 128;
local dropout = 0.2;
local lr = 0.1;
local max_filter_size = 4;
local num_filters = 128;
local output_dim = 128;
local ngram_filter_sizes = std.range(2, max_filter_size);
```

## After

```json
local embedding_dim = std.parseInt(std.extVar('embedding_dim'));
local dropout = std.parseJson(std.extVar('dropout'));
local lr = std.parseJson(std.extVar('lr'));
local max_filter_size = std.parseInt(std.extVar('max_filter_size'));
local num_filters = std.parseInt(std.extVar('num_filters'));
local output_dim = std.parseInt(std.extVar('output_dim'));
local ngram_filter_sizes = std.range(2, max_filter_size);
```

You can view a final configuration by clicking `details` below.

<details>

<br>

`imdb_optuna.jsonnet`

```json
local batch_size = 64;
local cuda_device = 0;
local num_epochs = 15;
local seed = 42;
local train_data_path = 'https://s3-us-west-2.amazonaws.com/allennlp/datasets/imdb/train.jsonl';
local validation_data_path = 'https://s3-us-west-2.amazonaws.com/allennlp/datasets/imdb/dev.jsonl';

local embedding_dim = std.parseInt(std.extVar('embedding_dim'));
local dropout = std.parseJson(std.extVar('dropout'));
local lr = std.parseJson(std.extVar('lr'));
local max_filter_size = std.parseInt(std.extVar('max_filter_size'));
local num_filters = std.parseInt(std.extVar('num_filters'));
local output_dim = std.parseInt(std.extVar('output_dim'));
local ngram_filter_sizes = std.range(2, max_filter_size);

{
  numpy_seed: seed,
  pytorch_seed: seed,
  random_seed: seed,
  dataset_reader: {
    type: 'text_classification_json',
    tokenizer: {
      type: 'whitespace',
    },
    token_indexers: {
      tokens: {
        type: 'single_id',
      },
    },
  },
  datasets_for_vocab_creation: ['train'],
  train_data_path: train_data_path,
  validation_data_path: validation_data_path,
  model: {
    type: 'basic_classifier',
    text_field_embedder: {
      token_embedders: {
        tokens: {
          embedding_dim: embedding_dim,
        },
      },
    },
    seq2vec_encoder: {
      type: 'cnn',
      embedding_dim: embedding_dim,
      ngram_filter_sizes: ngram_filter_sizes,
      num_filters: num_filters,
      output_dim: output_dim,
    },
    dropout: dropout,
  },
  data_loader: {
    batch_size: batch_size,
  },
  trainer: {
    cuda_device: cuda_device,
    num_epochs: num_epochs,
    optimizer: {
      lr: lr,
      type: 'sgd',
    },
    validation_metric: '+accuracy',
  },
}
```

</details>

</exercise>

<exercise id="4" title="Define search space">

Now that you have created the config, you can define the search space in Optuna.
Note that the parameter names are the same as those defined in the config earlier. The objective function is as follows.

```python
import optuna


def objective(trial: optuna.Trial) -> float:
    trial.suggest_int("embedding_dim", 32, 256)
    trial.suggest_int("max_filter_size", 2, 6)
    trial.suggest_int("num_filters", 32, 256)
    trial.suggest_int("output_dim", 32, 256)
    trial.suggest_float("dropout", 0.0, 0.8)
    trial.suggest_float("lr", 5e-3, 5e-1, log=True)

    executor = optuna.integration.allennlp.AllenNLPExecutor(
        trial=trial,  # trial object
        config_file="./config/imdb_optuna.jsonnet",  # path to jsonnet
        serialization_dir=f"./result/optuna/{trial.number}",
        metrics="best_validation_accuracy"
    )
    return executor.run()
```

Once we have defined the search space, we pass the trial object to AllenNLPExecutor.
It’s time to create executor!
AllenNLPExecutor takes a trial, a path to config, a path to snapshot, and a target metric to be optimized as input arguments (executor = AllenNLPExecutor(trial, config, snapshot, metric)).
Then let’s run executor.run to start optimization.
In each trial step in optimization, objective is called and does the following steps: (1) trains a model (2) gets a target metric on validation data (3) returns a target metric.


```python
if __name__ == '__main__':
    study = optuna.create_study(
        storage="sqlite:///result/trial.db",  # save results in DB
        sampler=optuna.samplers.TPESampler(seed=24),
        study_name="optuna_allennlp",
        direction="maximize",
    )

    timeout = 60 * 60 * 10  # timeout (sec): 60*60*10 sec => 10 hours
    study.optimize(
        objective,
        n_jobs=1,  # number of processes in parallel execution
        n_trials=30,  # number of trials to train a model
        timeout=timeout,  # threshold for executing time (sec)
    )

    optuna.integration.allennlp.dump_best_config(
        "./config/imdb_optuna.jsonnet",
        "best_imdb_optuna.json",
        study
    )
```

</exercise>

<exercise id="5" title="Writing your own script">
Additionally, you can use Optuna by writing your own script for creating a model and defining a search space.


[TODO: Installing Optuna is needed for executing this script.]
<codeblock source="part3/optuna/source" setup="part3/optuna/setup"></codeblock>

</exercise>
