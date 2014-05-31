__author__ = 'mdenil'


import os
import random
import cluster.job_manager
from cluster.config_utils import product, expand


def _sentence_embedding_size(k_max, n_feature_maps):
    return k_max * n_feature_maps


def _fully_connected_input_dimensions(k_max, n_feature_maps):
    return k_max * n_feature_maps


def make_job_template(
        embedding_dimension,
        n_feature_maps_w1,
        n_feature_maps_w2,
        n_feature_maps_d1,
        kernel_width_w1,
        kernel_width_w2,
        kernel_width_d1,
        k_max_w1,
        k_max_w2,
        k_max_d1,
        n_epochs,
        adagrad_gamma,
        validation_frequency,
        save_frequency,
        batch_size,
        walltime,
        n_classes,
        pmem,
        ppn,
        fully_connected_layer,
        regularizer,
        dropout_input,
        ):

    return {
        "templates": [

            {
                "target": "launcher.sh",
                "params_target": "params_launcher.yaml",
                "src": "launcher.sh",
                "params": {
                    "code_root": os.path.abspath(os.environ['HERE']),
                    "walltime": walltime,
                    "pmem": pmem,
                    "ppn": ppn,
                },
            },

            {
                "target": "train.py",
                "params_target": "params_train.yaml",
                "src": "train.py",
                "params": {
                    "train_data_json": os.path.abspath(
                        os.path.join(
                            os.environ['DATA'],
                            "stanfordmovie",
                            "stanfordmovie.train.sentences.clean.projected.json")),
                    "train_encoding_json": os.path.abspath(
                        os.path.join(
                            os.environ['DATA'],
                            "stanfordmovie",
                            "stanfordmovie.train.sentences.clean.dictionary.encoding.json")),
                    "n_validation": 500,
                    "batch_size": batch_size,
                    "fixed_n_sentences": 15,
                    "fixed_n_words": 50,
                    "regularizer": regularizer,
                    "adagrad_gamma": adagrad_gamma,
                    "n_epochs": n_epochs,
                    "validation_frequency": validation_frequency,
                    "save_frequency": save_frequency,
                },
            },

            {
                "target": "experiment_config.py",
                "params_target": "params_experiment_config.yaml",
                "src": "two_doc_layer_model.py",
                "params": {
                    "embedding_dimension": embedding_dimension,
                    "dropout_input": dropout_input,
                    "word_layers": [
                        {
                            # first word layer
                            "n_feature_maps": n_feature_maps_w1,
                            "kernel_width": kernel_width_w1,
                            "n_channels": embedding_dimension,
                            "k_pooling": k_max_w1,
                            "k_dynamic": 0.5,
                            "nonlinearity": "Tanh",
                        },
                        {
                            # second word layer
                            "n_feature_maps": n_feature_maps_w2,
                            "kernel_width": kernel_width_w2,
                            "n_channels": n_feature_maps_w1,
                            "k_pooling": k_max_w2,
                            "k_dynamic": -1,
                            "nonlinearity": "Tanh",
                        }
                    ],
                    "sentence_layers": [
                        {
                            # first sentence layer
                            "n_feature_maps": n_feature_maps_d1,
                            "kernel_width": kernel_width_d1,
                            "n_channels": _sentence_embedding_size(
                                k_max=k_max_w2,
                                n_feature_maps=n_feature_maps_w2),
                            "k_pooling": k_max_d1,
                            "k_dynamic": -1,
                            "nonlinearity": "Tanh",
                        },
                    ],
                    "fully_connected_layers": [
                        {
                            "n_input": _fully_connected_input_dimensions(
                                k_max=k_max_d1,
                                n_feature_maps=n_feature_maps_d1),

                            # output same size as input
                            "n_output": _fully_connected_input_dimensions(
                                k_max=k_max_d1,
                                n_feature_maps=n_feature_maps_d1),
                        },
                    ] if fully_connected_layer else [],
                    "n_classes": n_classes,
                    "softmax_input_dimensions": _fully_connected_input_dimensions(
                        k_max=k_max_d1,
                        n_feature_maps=n_feature_maps_d1),
                },
            }
        ],

        "task_params": {
            "launcher_file": "launcher.sh"
        },
    }


if __name__ == "__main__":
    experiment_dir = os.path.abspath(os.path.join(
        os.environ['RESULTS'],
        "two_doc_layer_model_gridsearch_2014-05-31"))

    parameter_search_grid = product([
        expand({
            "embedding_dimension": 20
        }),
        expand({
            "n_feature_maps_w1": 10,
            "n_feature_maps_w2": 20,
            "n_feature_maps_d1": 20,
        }),
        expand({
            "kernel_width_w1": 7,
            "kernel_width_w2": 5,
            "kernel_width_d1": 3,
        }),
        expand({
            "k_max_w1": 8,
            "k_max_w2": 6,
            "k_max_d1": 3,
        }),
        expand({
            "regularizer": [1e-4, 5e-4, 1e-3],
            "dropout_input": [True, False],
            "fully_connected_layer": [True, False],
            "n_epochs": 10000,
            "adagrad_gamma": [0.005, 0.01, 0.02],
            "validation_frequency": 50,
            "save_frequency": 50,
            "batch_size": 50,
            "walltime": "24:00:00",
            "n_classes": 2,
            "pmem": "6gb",
            "ppn": 1,
        })
    ])

    job_templates = [make_job_template(**params) for params in parameter_search_grid]

    print len(job_templates)
    #exit(0)

    jobs = []

    for job_id, task in enumerate(job_templates):
        jobs.append(cluster.job_manager.Job(
            job_id=job_id,
            base_dir=experiment_dir,
            params={},
            template_dir=os.path.join(os.environ['ROOT'], "experiment_templates"),
            tasks=[task],
            task_factory=cluster.job_manager.ClusterTask))

    # shuffle so that large tasks are spread out approximately uniformly
    random.shuffle(jobs)

    for job in jobs:
        for task in job.tasks():
            task.configure()
            task.launch()
