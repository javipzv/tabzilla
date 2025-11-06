import subprocess

def run_experiment(model_name, dataset_name):
    
    dataset_path = f".\\datasets\\{dataset_name}\\"
    command_template = [
        "python",
        ".\\tabzilla_experiment.py",
        "--experiment_config", ".\\tabzilla_experiment_config.yml",
        "--model_name", model_name,
        "--dataset_dir", dataset_path
    ]

    subprocess.run(command_template)

datasets = ['openml__credit-g__31', 'openml__electricity__219',
    'openml__elevators__3711', 'openml__nomao__9977', 'openml__profb__3561', 
    'openml__socmob__3797', 'openml__SpeedDating__146607', 'openml__ada_agnostic__3896',
    'openml__colic__25', 'openml__credit-approval__29', 'openml__heart-h__50', 
    'openml__jasmine__168911', 'openml__kc1__3917', 'openml__phoneme__9952', 
    'openml__qsar-biodeg__9957']

models = ['RandomForest', 'XGBoost', 'CatBoost', 'LightGBM',
          'EBMModel', 'GaussianNaiveBayesModel', 'LDAModel', 'NAMModel', 'LinearModelInterpret']

for i in range(len(datasets)):
    dataset = datasets[i]
    for j in range(len(models)):
        model = models[j]
        print(f"Running experiment for model: {model} on dataset: {dataset}")
        run_experiment(model, dataset)
