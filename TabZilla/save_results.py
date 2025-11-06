import os
import json
import numpy as np
import pandas as pd
import re

def save_results(results_dir):
    results = []

    for folder in os.listdir(results_dir):
        folder_path = os.path.join(results_dir, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith('.json'):
                    file_path = os.path.join(folder_path, file)
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        results.append(data)
    return results

# Specify your folder results here
dir = r""

results = save_results(dir)
results_df = pd.DataFrame(columns=['dataset_name', 'dataset_num_features', 'dataset_num_instances',
                                   'model_name', 'train_time', 'val_time', 'test_time',
                                   'train_log_loss', 'train_auc', 'train_accuracy', 'train_f1',
                                   'val_log_loss', 'val_auc', 'val_accuracy', 'val_f1',
                                   'test_log_loss', 'test_auc', 'test_accuracy', 'test_f1'])

for i in range(len(results)):
    r = results[i]

    try:

        dataset_name = r['dataset']['name']
        dataset_name = re.findall(r'openml__(.*)__\d+', dataset_name)[0]

        dataset_num_features = r['dataset']['num_features']
        dataset_num_instances = r['dataset']['num_instances']
        
        model_name = r['model']['name']

        train_time = round(np.mean(r['timers']['train']), 6)
        val_time = round(np.mean(r['timers']['val']), 6)
        test_time = round(np.mean(r['timers']['test']), 6)

        train_log_loss = round(np.mean(r['scorers']['train']['Log Loss']), 6)
        train_auc = round(np.mean(r['scorers']['train']['AUC']), 6)
        train_accuracy = round(np.mean(r['scorers']['train']['Accuracy']), 6)
        train_f1 = round(np.mean(r['scorers']['train']['F1']), 6)

        val_log_loss = round(np.mean(r['scorers']['val']['Log Loss']), 6)
        val_auc = round(np.mean(r['scorers']['val']['AUC']), 6)
        val_accuracy = round(np.mean(r['scorers']['val']['Accuracy']), 6)
        val_f1 = round(np.mean(r['scorers']['val']['F1']), 6)

        test_log_loss = round(np.mean(r['scorers']['test']['Log Loss']), 6)
        test_auc = round(np.mean(r['scorers']['test']['AUC']), 6)
        test_accuracy = round(np.mean(r['scorers']['test']['Accuracy']), 6)
        test_f1 = round(np.mean(r['scorers']['test']['F1']), 6)

        results_df.loc[i] = [dataset_name, dataset_num_features, dataset_num_instances, 
                            model_name, train_time, val_time, test_time,
                            train_log_loss, train_auc, train_accuracy, train_f1,
                            val_log_loss, val_auc, val_accuracy, val_f1,
                            test_log_loss, test_auc, test_accuracy, test_f1]
        
    except KeyError as e:
        print(dataset_name, model_name)
        continue
    
results_df.to_csv('results_see_differences.csv', index=False)