"""
Sensitivity Analysis

Description: This file is used for a sensitivity analysis on new parameters based on the Morris Sensitivity Analysis
Author: Lucas Brunschwig (lucas.brunschwig@gmail.com)

"""
# stdlib
import os
import random

# third party
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# autoprognosis absolute
from autoprognosis.explorers.core.selector import PipelineSelector
from autoprognosis.plugins.prediction import Predictions
from autoprognosis.utils.tester import classifier_metrics

from tmp_lucas import DataLoader

if __name__ == "__main__":

    run_analysis = False
    run_results = True
    n_runs = 5

    random_seeds = [0, 42, 100, 59, 74]

    multimodal_type = "image"
    classifier = "cnn_fine_tune"

    output = "sensitivity_results"

    output = output + "/" + classifier
    os.makedirs(output, exist_ok=True)

    if run_analysis:

        # ---------- PREPARING DATASET ---------- #
        print("Loading Datasets")
        DL = DataLoader(
            path_="../../data",
            data_src_="PAD-UFES",
            format_="PIL",
        )
        df_train, df_test = DL.load_dataset(
            sample=False, pacheco=False, full_size=False
        )
        print("Dataset Loaded")

        evaluator = classifier_metrics()

        targets_test = df_test[["label"]]
        targets_test.reset_index(inplace=True, drop=True)
        targets_train = df_train[["label"]]
        targets_train.reset_index(inplace=True, drop=True)
        encoder = LabelEncoder().fit(targets_train)
        targets_test = encoder.transform(targets_test.squeeze())
        targets_train = encoder.transform(targets_train.squeeze())

        df_train = df_train.drop(["label"], axis=1)
        df_test = df_test.drop(["label"], axis=1)

        # ---------- PREPARING PARAMETERS ---------- #

        df_dict = None
        if multimodal_type == "intermediate_fusion":
            X_train = {
                "img": df_train[["image"]],
                "tab": df_train[df_train.columns.difference(["image"])],
            }

            X_test = {
                "img": df_test[["image"]],
                "tab": df_test[df_train.columns.difference(["image"])],
            }
            y_train = targets_train
            y_test = targets_test

        elif multimodal_type == "image":

            X_train = df_train[["image"]]
            y_train = targets_train

            X_test = df_test[["image"]]
            y_test = targets_test

        # ---------- PREPARING PARAMETERS ---------- #

        if multimodal_type == "image":
            model = Predictions(category="classifier").get(classifier)
            params = model.hyperparameter_space_fqdn()

        elif multimodal_type == "intermediate_fusion":
            pipeline = PipelineSelector(
                classifier=classifier,
                imputers=["ice"],
                image_dimensionality_reduction=[],
                image_processing=[],
                fusion=[],
                feature_selection=[],
                feature_scaling=[],
            )
            model = pipeline.get_multimodal_pipeline_from_named_args(**{})
            params = model.hyperparameter_space()[classifier]

        params_dict = {}
        for param in params:
            params_dict[param.name.split(".")[-1]] = param.choices

        if multimodal_type == "intermediate_fusion":
            params_dict["conv_name"] = "alexnet"
            params_dict["n_units_hidden"] = 75

        else:
            params_dict["conv_net"] = "alexnet"

        random_param_selection = {}

        # storage
        evaluation_aucroc = {
            name_: {choice: [] for choice in choices_}
            for name_, choices_ in params_dict.items()
            if isinstance(choices_, list)
            if name_ in ["n_unfrozen_layers", "dropout"]
        }
        evaluation_accuracy = {
            name_: {choice: [] for choice in choices_}
            for name_, choices_ in params_dict.items()
            if isinstance(choices_, list)
            if name_ in ["n_unfrozen_layers", "dropout"]
        }
        evaluation_balanced = {
            name_: {choice: [] for choice in choices_}
            for name_, choices_ in params_dict.items()
            if isinstance(choices_, list)
            if name_ in ["n_unfrozen_layers", "dropout"]
        }

        for i in range(n_runs):

            random.seed(random_seeds[i])

            # For each run choose a set of random param
            for name, choices in params_dict.items():
                if multimodal_type == "intermediate_fusion":
                    name_arg = f"prediction.classifier.{classifier}." + name
                else:
                    name_arg = name
                if isinstance(choices, list):
                    value = random.choice(choices)
                    random_param_selection[name_arg] = value
                else:
                    random_param_selection[name_arg] = choices

            print(f"Run {i}: {random_param_selection}")

            # Select one parameter and evaluate the model with random set up to see how
            for current_name, current_choices in params_dict.items():
                if not isinstance(current_choices, list) and current_name not in [
                    "n_unfrozen_layers",
                    "dropout",
                ]:
                    continue
                if multimodal_type == "intermediate_fusion":
                    current_name_arg = (
                        f"prediction.classifier.{classifier}." + current_name
                    )
                else:
                    current_name_arg = current_name

                # Evaluate all params value with this params
                previous_choice = random_param_selection[current_name_arg]
                previous_current_name = current_name_arg

                for choice in current_choices:
                    random_param_selection[current_name_arg] = choice
                    print(current_name_arg, choice)
                    if multimodal_type == "image":
                        model = Predictions(category="classifier").get(
                            classifier, **random_param_selection
                        )
                        model.fit(X_train, y_train)
                        preds = model.predict_proba(X_test)
                    elif multimodal_type == "intermediate_fusion":
                        pipeline = PipelineSelector(
                            classifier=classifier,
                            imputers=["ice"],
                            image_dimensionality_reduction=[],
                            image_processing=[],
                            fusion=[],
                            feature_selection=[],
                            feature_scaling=[],
                            multimodal_type=multimodal_type,
                        )

                        model = pipeline.get_multimodal_pipeline_from_named_args(
                            **random_param_selection
                        )
                        model.fit(X_train, y_train)
                        preds = model.predict_proba(X_test)

                    results = evaluator.score_proba(y_test, preds)

                    evaluation_aucroc[current_name][choice].append(results["aucroc"])
                    evaluation_accuracy[current_name][choice].append(
                        results["accuracy"]
                    )
                    evaluation_balanced[current_name][choice].append(
                        results["balanced_accuracy"]
                    )

                # reset to the previous state to keep same initial params
                random_param_selection[previous_current_name] = previous_choice

        for name_ in evaluation_accuracy.keys():
            # Save to csv for each param
            param_summary = pd.DataFrame.from_dict(evaluation_accuracy[name_])
            param_summary.to_csv(f"{output}/{name_}_accuracy.csv")

            param_summary = pd.DataFrame.from_dict(evaluation_aucroc[name_])
            param_summary.to_csv(f"{output}/{name_}_aucroc.csv")

            param_summary = pd.DataFrame.from_dict(evaluation_balanced[name_])
            param_summary.to_csv(f"{output}/{name_}_balanced.csv")
        # ---------- PREPARING PARAMETERS ---------- #

        # ---------- RUNNING SENSITIVITY ANALYSIS ---------- #

    if run_results:

        accuracy_summary = {}

        # Load Parameters Results
        for file in os.listdir(output):
            if (
                file.endswith(".csv")
                and "accuracy" in file
                and "balanced_accuracy" not in file
            ):
                accuracy_summary["_".join(file.split("_")[:-1])] = pd.read_csv(
                    os.path.join(output, file), index_col=0
                )

        if multimodal_type == "image":
            fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(30, 15))
        elif multimodal_type == "intermediate_fusion":
            fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(30, 20))

        line_color = "red"  # Color for the line plot

        for i, ((name, df), ax) in enumerate(
            zip(accuracy_summary.items(), axes.ravel()[:-1])
        ):

            # Scatter plot with legend=False
            for index, row in df.T.iterrows():
                sns.scatterplot(
                    x=df.T.columns,
                    y=row,
                    label=index,
                    ax=ax,
                    marker="x",
                    s=100,
                    legend=False,
                )

            ax.set_ylim(0, 1.0)
            ax2 = ax.twinx()

            lines1, labels1 = ax.get_legend_handles_labels()

            # Line plot with legend=False and specified color
            sns.lineplot(
                x=df.T.columns,
                y=df.T.max(axis=0) - df.T.min(axis=0),
                color=line_color,
                alpha=0.5,
                label="Max-Min",
                ax=ax2,
                legend=False,
            )

            # Set y-axis limits
            ax2.set_ylim(0, 0.5)

            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

            # Display ticks as crosses on the primary y-axis
            ax.tick_params(
                axis="y", direction="inout", length=10, width=2, grid_alpha=0.5
            )
            ax.tick_params(
                axis="x", direction="inout", length=10, width=2, grid_alpha=0.5
            )
            ax2.tick_params(
                axis="y",
                direction="inout",
                length=10,
                width=2,
                grid_alpha=0.5,
                colors=line_color,
            )  # Match tick colors with line color

            lines2, labels2 = ax2.get_legend_handles_labels()

            # Manually create a combined legend
            ax.legend(lines1, labels1, loc=0, fontsize=14)
            ax.set_title(name, fontsize=18)
            ax.grid()
            ax2.grid()

            # Set y-axis labels based on subplot position
            if i % 4 == 0:  # leftmost subplot of each row
                ax.set_ylabel(
                    "accuracy", fontsize=18
                )  # Match label color with line color
                ax2.set_ylabel("")
                ax2.set_yticklabels([])
            elif i % 4 == 3:  # rightmost subplot of each row
                ax2.set_ylabel("max - min difference", fontsize=18, color=line_color)
                ax.set_ylabel("")
                ax.set_yticklabels([])

            else:  # middle subplot
                ax.set_yticklabels([])
                ax2.set_yticklabels([])

                ax.set_ylabel("")
                ax2.set_ylabel("")

            ax.set_xlabel("run number")
        fig.suptitle("Sensitivity Analysis CNN fine-tuning", size=18)
        plt.savefig(output + f"/sensitivity_analysis_{classifier}.png")
