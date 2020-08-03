---
title: 'Hyperparameter Optimization with Optuna'
description: "This chapter gives a basic tutorial for optimizing hyperparameters of your model."
type: chapter
---

<textblock>

In this chapter, we'll give a quick tutorial of hyperparameter optimization for your AllenNLP model.

</textblock>

<exercise id="1" title="Hyperparameter matters!">

A choice of hyperparameters sometimes has a strong impact to the performance of a model.
Even if you use the same model, performance could drastically change by the hyperparameters (e.g. learning rate, dimensionality of word embeddings) you use.
Following figure shows the performance differences with changing hyperparameters.
<img src="/part3/hyperparameter-optimization-with-optuna/hyperparameter_matters.jpg" alt="Hyperparameter matters!" />

The choice of hyperparameters is often based on heuristics of experts, or random search, and may complicate the code.
<a href="https://optuna.org">Optuna</a> is the library for hyperparameter optimization, which allow users to optimize hyperparameters automatically+easily.
Various sophisticated algorithms for searching parameters such as [Tree-structured Parzen Estimator](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization) and
[CMA Evolution Strategy](https://arxiv.org/abs/1604.00772), and pruning unpromissing trainings such as [Hyperband](http://jmlr.org/papers/v18/16-558.html) are implemented.

</exercise>

<exercise id="2" title="Building your model">

This tutorial works on sentiment analysis, one kind of text classification.
We use IMDb review dataset, which contains 20,000 positive/negative reviews for training and 5,000 reviews for validating the performance of model.
If you haven't read <a href="https://guide.allennlp.org/your-first-model#1">the previous tutorial for text classification</a>, it would help you.

I show a sample configuration of a CNN-based classifier below.
Note that this model has 6 hyperparameters: `embedding_dim`, `dropout`, `lr`, `max_filter_size`, `num_filters`, and `output_dim`.

```json
// imdb_baseline.jsonnet

local batch_size = 64;
local cuda_device = 0;
local num_epochs = 15;
local seed = 42;
local train_data_path = 'https://s3-us-west-2.amazonaws.com/allennlp/datasets/imdb/train.jsonl';
local validation_data_path = 'https://s3-us-west-2.amazonaws.com/allennlp/datasets/imdb/dev.jsonl';

// hyperparameters
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

I selected these hyperparameters simply based on the huristics.
Of course, we can train this model using AllenNLP CLI.
I run `allennlp train imdb_baseline.jsonnet` five times with changing random seeds.
As the result, the average of validation accuracy was 0.828 (±0.004).

</exercise>

<exercise id="3" title="Masking Hyperparameters for Optuna">

Let's dive into hyperparameter optimization.
Optuna offers a integration for AllenNLP, named <a href="https://optuna.readthedocs.io/en/stable/reference/integration.html#optuna.integration.AllenNLPExecutor">`AllenNLPExecutor`</a>.
We can use `AllenNLPExecutor` by following steps: `Masking parameters` and `Defining search space`.

First, we replace values of hyperparameters with `std.extVar` for tell Optuna what parameters to be optimized.
Remember that call `std.parseInt` or `std.parseFloat` for numerical parameters.

### Before

```json
local embedding_dim = 128;
local dropout = 0.2;
local lr = 0.1;
local max_filter_size = 4;
local num_filters = 128;
local output_dim = 128;
local ngram_filter_sizes = std.range(2, max_filter_size);
```

### After

```json
local embedding_dim = std.parseInt(std.extVar('embedding_dim'));
local dropout = std.parseJson(std.extVar('dropout'));
local lr = std.parseJson(std.extVar('lr'));
local max_filter_size = std.parseInt(std.extVar('max_filter_size'));
local num_filters = std.parseInt(std.extVar('num_filters'));
local output_dim = std.parseInt(std.extVar('output_dim'));
local ngram_filter_sizes = std.range(2, max_filter_size);
```

That's all. You can view a final configuration by clicking `details` below.

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

<exercise id="4" title="Defining Search Space for Hyperparameters in Pythonic way">

Now that you have created the config, next step is defining search spaces of hyperparameters.
In Optuna, it defines a search space by creating `objective function`.
Each search space for hyperparameter is declared by `suggest_int` and `suggest_float`.
If you want to optimize categorical hyperparameters, you can use `suggest_categorical`.
Please see <a href="https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial">our documentation</a> for more information.
In this tutorial, I create a objective function like following (Note that the names of hyperparameter should be the same as those defined in the configuration earlier):

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

After defining search spaces using `trial.suggest_int` and `trial.suggest_float`, `trial` should be passed to `AllenNLPExecutor`.
`AllenNLPExecutor` takes three required arguments; `trial` (Optuna's object), `config_file` (path to a model configuration),
`serialization_dir` (directory for saving model snapshot, log, etc.), and `metrics` you want to optimize.
In above example, we create an instance of `AllenNLPExecutor` as `executor`.
Once creating `executor`, it starts a training by `executor.run()`.
In each trial step in optimization, objective is called and does the following steps:

1. Train a model (`executor.run()`)
2. Get and return a target metric on validation data (`executor.run()` returns a value of specified metric)

Now, we finished defining objective function. :tada:
Let's write Optuna stufs for launching optimization!
In Optuna, we create a study object and pass the objective function to method `optimize()` as follows.
You can specify something; a way to save a result of optimization, a sampler for searching hyperparameters (TPESampler is based on Bayesian Optimization),
direction for optimizing (maximize or minimize), and number of jobs for distributed training, and timeout.

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
```

</exercise>

<exercise id="5" title="Results of Hyperparameter Optimization">

If `study.optimize` successfully runs, `trial.db` would be created in the directory `result`.
[Tips] You can change a names of database file and directory by changing the value of `storage` in `optuna.create_study`.

You can see and analyze a result by passing `study` object to various methods implemented in Optuna.
If you want to separate an analysis from optimization, you can save `study` in storages (e.g. RDB) and load it in another script.

```python
study = optuna.load_study(
  storage="sqlite:///result/trial.db",
  study_name="optuna_allennlp"
)
```

First of all, let's check results of each trials with a `pandas` dataframe.

```python
study.trials_dataframe()
```

<img src="/part3/hyperparameter-optimization-with-optuna/trials_dataframe.jpg" alt="Dataframe">

Next, visualize a history of optimization.
To plot a history of optimization, we can use `optuna.visualization.plot_optimization_history`.
I also put a validation accuracy of baseline model as a reference.
It shows that Optuna successfully find hyperparameters for achieving better performance.
Note that this figure shows one result of optimizations.
As the case of baseline, I performed optimization five times with changing random seeds and got the averaged validation accuracy 0.909 (±0.004), which outperforms the baseline with large margine.

```python
optuna.visualization.plot_optimization_history(study)
```

<br>
<img src="/part3/hyperparameter-optimization-with-optuna/optimization_history.jpg" alt="Plot Optimization History" />

From the recent release of Optuna, we can evaluate parameter importances based on finished trials.
There are two evaluators available: [fANOVA](http://proceedings.mlr.press/v32/hutter14.html) and [MDI](https://papers.nips.cc/paper/4928-understanding-variable-importances-in-forests-of-randomized-trees).
To show importances of hyperparameters, it uses `optuna.visualization.plot_param_importances`:

```python
optuna.visualization.plot_param_importances(study)
```

In this plot, we can see that `lr` is the most important in this experiment.

<br>
<img src="/part3/hyperparameter-optimization-with-optuna/hyperparameter_importance.jpg" alt="Plot Hyperparameter Importance" />

Additionally, you can export a configuration with optimized hyperparameters.

```python
dump_best_config("./imdb_optuna.jsonnet", "./best_config.json", study)
```

It will create a configuration named `best_config.json`.
If you want to retrain a model with the best hyperparameters, it would be helpful.

</exercise>

<exercise id="6" title="[Advanced] Writing your own script">
Additionally, you can use Optuna by writing your own script for creating a model and defining a search space.

You may notice that `AllenNLPPruningCallback` is specified in `epoch_callbacks`, it is our recent work on Optuna x AllenNLP integration.
Using `AllenNLPPruningCallback`, you can use efficient pruning algorithms such as <a href="http://jmlr.org/papers/v18/16-558.html">Hyperband</a> in training a model.
<!-- [TODO: Installing Optuna is needed for executing this script.] -->
<codeblock source="part3/optuna/source" setup="part3/optuna/setup"></codeblock>

</exercise>
