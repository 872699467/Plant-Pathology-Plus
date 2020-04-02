import pandas as pd
import csv

if __name__ == '__main__':
    squeezenet_res = pd.read_csv('results/resnet_results.csv').iloc[:, [1, 2, 3, 4]].values
    densenet_res = pd.read_csv('results/densenet_results.csv').iloc[:, [1, 2, 3, 4]].values
    sub = pd.read_csv('data/sample_submission.csv')
    ensemble_1, ensemble_2, ensemble_3 = [sub] * 3
    ensemble_1.iloc[:, [1, 2, 3, 4]] = 0.25 * squeezenet_res + 0.75 * densenet_res
    ensemble_2.iloc[:, [1, 2, 3, 4]] = 0.5 * squeezenet_res + 0.5 * densenet_res
    ensemble_3.iloc[:, [1, 2, 3, 4]] = 0.75 * squeezenet_res + 0.25 * densenet_res

    ensemble_1.to_csv('results/submission_ensemble_1.csv', index=False)
    ensemble_2.to_csv('results/submission_ensemble_2.csv', index=False)
    ensemble_3.to_csv('results/submission_ensemble_3.csv', index=False)
    print('finish')
