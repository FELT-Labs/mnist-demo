from data import transform_cnn, transform_flat

experiments = [
    {
        "iterations": 1,
        "full_iterations": 1,
        "partitions": 3,
        "data_transform": transform_flat,
        "model_definition": {
            "model_type": "sklearn",
            "model_name": "LogisticRegression",
            "init_params": {"max_iter": 100},
        },
    },
    {
        "iterations": 1,
        "full_iterations": 1,
        "partitions": 3,
        "data_transform": transform_flat,
        "model_definition": {
            "model_type": "sklearn",
            "model_name": "NearestCentroidClassifier",
            "init_params": {},
        },
    },
    {
        "iterations": 8,
        "full_iterations": 8,
        "partitions": 3,
        "data_transform": transform_flat,
        "model_definition": {
            "model_type": "sklearn",
            "model_name": "MLPClassifier",
            "init_params": {
                "hidden_layer_sizes": [50, 50],
                "max_iter": 50,
                "warm_start": True,
                "random_state": 10,
            },
        },
    },
    {
        "iterations": 6,
        "full_iterations": 6,
        "partitions": 3,
        "data_transform": transform_cnn,
        "model_definition": {
            "model_type": "tensorflow",
            "model_name": "EfficientNetB0",
            "init_params": {
                "input_shape": [32, 32, 3],
                "classes": 10,
                "include_top": True,
                "weights": None,
            },
        },
        "fit_args": {"epochs": 1, "batch_size": 32},
    },
    {
        "iterations": 10,
        "full_iterations": 10,
        "partitions": 3,
        "data_transform": lambda data: data.reshape((len(data), 28, 28, 1)) / 255.5,
        "model_definition": {
            "model_type": "tensorflow",
            "model_name": "CustomCNN",
            "init_params": {
                "input_shape": [28, 28, 1],
                "classes": 10,
                "architecture": "C-20-3-2-same,M-3-2,F,H-100",
            },
        },
        "fit_args": {"epochs": 1, "batch_size": 32},
    },
]
