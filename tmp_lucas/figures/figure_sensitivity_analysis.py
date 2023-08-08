"""
Sensitivity Analysis

Description: This file is used for a sensitivity analysis on new parameters
Author: Lucas Brunschwig (lucas.brunschwig@gmail.com)

"""
# stdlib
import os
import random

# third party
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# autoprognosis absolute
from autoprognosis.plugins.prediction import Predictions
from autoprognosis.utils.tester import evaluate_estimator, evaluate_multimodal_estimator

from tmp_lucas import DataLoader

if __name__ == "__main__":

    output = "sensivity"
    os.makedirs(output, exist_ok=True)

    run_analysis = True
    n_runs = 10
    analyze_results = True

    multimodal_type = "image"
    classifier = "cnn_fine_tune"

    if run_analysis:
        print("Loading Datasets")
        DL = DataLoader(
            path_="../../data",
            data_src_="PAD-UFES",
            format_="PIL",
        )
        df = DL.load_dataset()
        df = df.sample(n=1000, random_state=42)
        print("Dataset Loaded")

        targets = df[["label"]]
        targets.reset_index(inplace=True, drop=True)
        targets = LabelEncoder().fit_transform(targets)

        df.drop(["label"], axis=1)

        df_dict = None
        if multimodal_type == "intermediate_fusion":
            df_dict = {
                "img": df[["image"]],
                "tab": df[df.columns.difference(["image"])],
            }
            df_dict["img"].reset_index(inplace=True, drop=True)
            df_dict["tab"].reset_index(inplace=True, drop=True)
            print("Extracting Parameters")

        elif multimodal_type == "image":
            df_dict = df[["image"]]
            df_dict.reset_index(inplace=True, drop=True)

        model = Predictions(category="classifier").get(classifier)

        params = model.hyperparameter_space_fqdn()
        params_dict = {}
        for param in params:
            params_dict[param.name.split(".")[-1]] = param.choices

        # Fix Parameters
        params_dict["conv_net"] = "alexnet"
        params_dict["aucroc"] = 0
        params_dict["accuracy"] = 0
        params_dict["balanced_accuracy"] = 0

        initial_params = {}
        for name, choice in params_dict.items():
            if isinstance(choice, list):
                initial_params[name] = random.choice(choice)
            else:
                initial_params[name] = params_dict[name]

        print("Running sensitivity analysis...")
        param_runs = pd.DataFrame(columns=list(params_dict.keys()))
        for i in range(n_runs):

            model = Predictions(category="classifier").get(classifier, **initial_params)

            if multimodal_type == "intermediate_fusion":
                results = evaluate_multimodal_estimator(
                    model,
                    X=df_dict,
                    Y=targets,
                    multimodal_type="late_fusion",
                    n_folds=5,
                )
            elif multimodal_type == "image":
                results = evaluate_estimator(
                    model,
                    X=df_dict,
                    Y=targets,
                    n_folds=5,
                )

            initial_params["aucroc"] = results["str"]["aucroc"]
            initial_params["accuracy"] = results["str"]["accuracy"]
            initial_params["balanced_accuracy"] = results["str"]["balanced_accuracy"]
            param_runs.loc["initial_params_" + str(i)] = initial_params

            for name, choice in params_dict.items():
                if not isinstance(choice, list):
                    continue
                new_param = random.choice(choice)
                count = 0
                count_max = 20
                tested_params = np.unique(param_runs[name].to_numpy())

                # Try to find a new parameters
                while count < count_max and new_param in tested_params:
                    new_param = random.choice(choice)
                    count += 1

                # If no new parameters possible just choose another one
                if new_param == initial_params[name]:
                    while new_param == initial_params[name]:
                        new_param = random.choice(choice)

                initial_params[name] = new_param
                model = Predictions(category="classifier").get(
                    classifier, **initial_params
                )

                if multimodal_type == "intermediate_fusion":
                    results = evaluate_multimodal_estimator(
                        model,
                        X=df_dict,
                        Y=targets,
                        multimodal_type="late_fusion",
                        n_folds=5,
                    )
                elif multimodal_type == "image":
                    results = evaluate_estimator(
                        model,
                        X=df_dict,
                        Y=targets,
                        n_folds=5,
                    )

                initial_params["aucroc"] = results["str"]["aucroc"]
                initial_params["accuracy"] = results["str"]["accuracy"]
                initial_params["balanced_accuracy"] = results["str"][
                    "balanced_accuracy"
                ]
                param_runs.loc[name + f"_{i}"] = initial_params

        param_runs.to_csv(f"{output}/sensitivity_analysis.csv")

    if analyze_results:
        results = pd.read_csv("tmp/sensitivity_analysis.csv")
        results.drop("Unnamed: 0", axis=1, inplace=True)
        n_params = len(results) // 5
        params_name = results.columns
        previous_accuracy = None
        sensitivity_analysis = {column: [] for column in params_name[:-3]}
        for ix, row in results.iterrows():
            column = ix % n_params
            accuracy = row.iloc[-2]
            if column > 0:
                sensitivity_analysis[results.columns[column - 1]].append(
                    float(accuracy.split(" ")[0])
                    - float(previous_accuracy.split(" ")[0])
                )
            previous_accuracy = accuracy

        params = list(sensitivity_analysis.keys())
        importance_mean = []
        importance_std = []
        for values in sensitivity_analysis.values():
            importance_mean.append(np.mean(np.abs(values)))
            importance_std.append(np.std(np.abs(values)))
        fig, ax = plt.subplots(figsize=(18, 18))
        ax.set_xticks(rotation=90)  # Plot the bars for the mean values
        ax.bar(params, importance_mean, yerr=importance_std, capsize=5)

        # Set labels and title
        ax.set_xlabel("Parameters name")
        ax.set_ylabel("Morris Sensitivity")
        ax.set_title("Morris Sensitivity Analysis for Intermediate Fusion")

        # Display the plot
        plt.tight_layout()
        plt.savefig(f"{output}/sensitivity_analysis.png", bbox_inches="tight")
