# stdlib
from pathlib import Path
import time
from typing import Any, List, Optional, Tuple

# third party
import numpy as np
import pandas as pd

# autoprognosis absolute
from autoprognosis.exceptions import StudyCancelled
from autoprognosis.explorers.core.defaults import (
    default_classifiers_names,
    default_feature_scaling_names,
    default_feature_selection_names,
    default_fusion,
    default_image_classsifiers_names,
    default_image_dimensionality_reduction,
    default_image_processing,
    default_multimodal_names,
)
from autoprognosis.explorers.multimodal_classifiers_combos import (
    MultimodalEnsembleSeeker,
)
from autoprognosis.hooks import DefaultHooks, Hooks
import autoprognosis.logger as log
from autoprognosis.studies._base import Study
from autoprognosis.utils.distributions import enable_reproducible_results
from autoprognosis.utils.serialization import (
    dataframe_hash,
    load_model_from_file,
    save_model_to_file,
)
from autoprognosis.utils.tester import evaluate_multimodal_estimator

PATIENCE = 10
SCORE_THRESHOLD = 0.65


class MultimodalStudy(Study):
    """
    Core logic for Multimodal studies.
    A study automatically handles imputation, preprocessing and model selection for a certain dataset.
    The output is an optimal model architecture, selected by the AutoML logic.
    Args:
        dataset: DataFrame.
            The multimodal dataset to analyze. Currently, support clinical data with image.
        image: str.
            The image column in the dataset
        target: str.
            The target column in the dataset.
        num_iter: int.
            Maximum Number of optimization trials. This is the limit of trials for each base estimator in the "classifiers" list, used in combination with the "timeout" parameter. For each estimator, the search will end after "num_iter" trials or "timeout" seconds.
        num_study_iter: int.
            The number of study iterations. This is the limit for the outer optimization loop. After each outer loop, an intermediary model is cached and can be used by another process, while the outer loop continues to improve the result.
        timeout: int.
            Maximum wait time(seconds) for each estimator hyperparameter search. This timeout will apply to each estimator in the "classifiers" list.
        metric: str.
            The metric to use for optimization.
            Available objective metrics:
                - "aucroc" : the Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
                - "aucprc" : The average precision summarizes a precision-recall curve as the weighted mean of precisions achieved at each threshold, with the increase in recall from the previous threshold used as the weight.
                - "accuracy" : Accuracy classification score.
                - "f1_score_micro": F1 score is a harmonic mean of the precision and recall. This version uses the "micro" average: calculate metrics globally by counting the total true positives, false negatives and false positives.
                - "f1_score_macro": F1 score is a harmonic mean of the precision and recall. This version uses the "macro" average: calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
                - "f1_score_weighted": F1 score is a harmonic mean of the precision and recall. This version uses the "weighted" average: Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).
                - "mcc": The Matthews correlation coefficient is used in machine learning as a measure of the quality of binary and multiclass classifications. It takes into account true and false positives and negatives and is generally regarded as a balanced measure which can be used even if the classes are of very different sizes.
                - "kappa", "kappa_quadratic":  computes Cohen’s kappa, a score that expresses the level of agreement between two annotators on a classification problem.
        study_name: str.
            The name of the study, to be used in the caches.
        feature_scaling: list.
            Plugin search pool to use in the pipeline for scaling. Defaults to : ['maxabs_scaler', 'scaler', 'feature_normalizer', 'normal_transform', 'uniform_transform', 'nop', 'minmax_scaler']
            Available plugins, retrieved using `Preprocessors(category="feature_scaling").list_available()`:
                - 'maxabs_scaler'
                - 'scaler'
                - 'feature_normalizer'
                - 'normal_transform'
                - 'uniform_transform'
                - 'nop' # empty operation
                - 'minmax_scaler'
        feature_selection: list.
            Plugin search pool to use in the pipeline for feature selection. Defaults ["nop", "variance_threshold", "pca", "fast_ica"]
            Available plugins, retrieved using `Preprocessors(category="dimensionality_reduction").list_available()`:
                - 'feature_agglomeration'
                - 'fast_ica'
                - 'variance_threshold'
                - 'gauss_projection'
                - 'pca'
                - 'nop' # no operation
        image_preprocessing: list.
            Plugin search pool to use in the pipeline for optimal preprocessing. If the list is empty, the program assumes that
            you preprocessed the images yourself.
        classifiers: list.
            Plugin search pool to use in the pipeline for prediction. Defaults to ["random_forest", "xgboost", "logistic_regression", "catboost"].
            Available plugins, retrieved using `Classifiers().list_available()`:
                - 'adaboost'
                - 'bernoulli_naive_bayes'
                - 'neural_nets'
                - 'linear_svm'
                - 'qda'
                - 'decision_trees'
                - 'logistic_regression'
                - 'hist_gradient_boosting'
                - 'extra_tree_classifier'
                - 'bagging'
                - 'gradient_boosting'
                - 'ridge_classifier'
                - 'gaussian_process'
                - 'perceptron'
                - 'lgbm'
                - 'catboost'
                - 'random_forest'
                - 'tabnet'
                - 'multinomial_naive_bayes'
                - 'lda'
                - 'gaussian_naive_bayes'
                - 'knn'
                - 'xgboost'
        imputers: list.
            Plugin search pool to use in the pipeline for imputation. Defaults to ["mean", "ice", "missforest", "hyperimpute"].
            Available plugins, retrieved using `Imputers().list_available()`:
                - 'sinkhorn'
                - 'EM'
                - 'mice'
                - 'ice'
                - 'hyperimpute'
                - 'most_frequent'
                - 'median'
                - 'missforest'
                - 'softimpute'
                - 'nop'
                - 'mean'
                - 'gain'
        hooks: Hooks.
            Custom callbacks to be notified about the search progress.
        workspace: Path.
            Where to store the output model.
        score_threshold: float.
            The minimum metric score for a candidate.
        id: str.
            The id column in the dataset.
        random_state: int
            Random seed
        sample_for_search: bool
            Subsample the evaluation dataset in the search pipeline. Improves the speed of the search.
        max_search_sample_size: int
            Subsample size for the evaluation dataset, if `sample` is True.
        n_folds_cv: int.
            Number of cross-validation folds to use for study evaluation
        ensemble_size: int
            Maximum number of models to include in the ensemble
    Example:
        >>> from sklearn.datasets import load_breast_cancer
        >>>
        >>> from autoprognosis.studies.classifiers import ClassifierStudy
        >>> from autoprognosis.utils.serialization import load_model_from_file
        >>> from autoprognosis.utils.tester import evaluate_estimator
        >>>
        >>> X, Y = load_breast_cancer(return_X_y=True, as_frame=True)
        >>>
        >>> df = X.copy()
        >>> df["target"] = Y
        >>>
        >>> study_name = "example"
        >>>
        >>> study = MultimodalStudy(
        >>>     study_name=study_name,
        >>>     dataset=df,  # pandas DataFrame
        >>>     target="target",  # the label column in the dataset
        >>>     image='image',  # the image column in the dataset
        >>> )
        >>> model = study.fit()
        >>>
        >>> # Predict the probabilities of each class using the model
        >>> model.predict_proba(X)
    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        image: str,
        target: str,
        multimodal_type: str,
        preprocess_images: bool = True,
        num_iter: int = 20,
        num_study_iter: int = 10,
        num_ensemble_iter: int = 15,
        timeout: Optional[int] = 360,
        metric: str = "aucroc",
        study_name: Optional[str] = None,
        feature_scaling: List[str] = default_feature_scaling_names,
        feature_selection: List[str] = default_feature_selection_names,
        image_processing: List[str] = default_image_processing,
        image_dimensionality_reduction: List[str] = [],
        fusion: List[str] = [],
        classifiers: List[str] = [],
        image_classifiers: List[str] = [],
        imputers: List[str] = ["ice"],
        workspace: Path = Path("tmp"),
        hooks: Hooks = DefaultHooks(),
        score_threshold: float = SCORE_THRESHOLD,
        group_id: Optional[str] = None,
        nan_placeholder: Any = None,
        random_state: int = 0,
        sample_for_search: bool = True,
        max_search_sample_size: int = 10000,
        ensemble_size: int = 3,
        n_folds_cv: int = 5,
    ) -> None:
        super().__init__()

        enable_reproducible_results(random_state)

        self.hooks = hooks
        dataset = pd.DataFrame(dataset)

        if multimodal_type not in [
            "early_fusion",
            "intermediate_fusion",
            "late_fusion",
        ]:
            raise ValueError(
                "multimodal_type expect one of the three values "
                "(early_fusion, intermediate_fusion, late_fusion)"
            )
        self.multimodal_type = multimodal_type

        if not preprocess_images:
            image_processing = []

        # Early Fusion requires image dimensionality reduction and fusion plugins
        if multimodal_type == "early_fusion":
            if not image_dimensionality_reduction:
                image_dimensionality_reduction = default_image_dimensionality_reduction
            if not fusion:
                fusion = default_fusion

        if (
            multimodal_type == "intermediate_fusion"
            and not image_dimensionality_reduction
        ):
            image_dimensionality_reduction = default_image_dimensionality_reduction

        # Fusion is only used in early fusion
        if multimodal_type != "early_fusion":
            if fusion:
                fusion = []
                log.warning(
                    "Fusion plugin are only included in early fusion - multimodal_type"
                )
            if (
                multimodal_type != "intermediate_fusion"
                and image_dimensionality_reduction
            ):
                image_dimensionality_reduction = []
                log.warning(
                    "Image dimensionality reduction plugin is not used in late fusion"
                )

        if not classifiers:
            if multimodal_type == "early_fusion":
                classifiers = default_classifiers_names
            elif multimodal_type == "intermediate_fusion":
                classifiers = default_multimodal_names
            elif multimodal_type == "late_fusion":
                classifiers = default_classifiers_names
        if multimodal_type == "late_fusion" and not image_classifiers:
            image_classifiers = default_image_classsifiers_names

        if nan_placeholder is not None:
            dataset = dataset.replace(nan_placeholder, np.nan)

        if dataset.isnull().values.any():
            if len(imputers) == 0:
                raise RuntimeError("Please provide at least one imputation method")
        else:
            imputers = []

        # Sort modalities
        self.multimodal_key = {}
        non_tabular_column = []
        if image is not None:
            if not isinstance(image, list):
                image = [image]
            non_tabular_column.extend(image)
            self.multimodal_key["img"] = image

        self.multimodal_key["tab"] = dataset.columns.difference(
            non_tabular_column + [target]
        )

        drop_cols = [target]
        self.group_ids = None
        if group_id is not None:
            drop_cols.append(group_id)
            self.group_ids = dataset[group_id]

        self.Y = dataset[target]
        self.X = dataset.drop(columns=drop_cols)

        if sample_for_search:
            sample_size = min(len(self.Y), max_search_sample_size)

            counts = self.Y.value_counts().to_dict()
            weights = self.Y.apply(lambda s: counts[s])
            self.search_Y = self.Y.sample(
                sample_size, random_state=random_state, weights=weights
            )
            self.search_X = self.X.loc[self.search_Y.index].copy()
            self.search_group_ids = None
            if self.group_ids:
                self.search_group_ids = self.group_ids.loc[self.search_Y.index].copy()
        else:
            self.search_X = self.X.copy()
            self.search_Y = self.Y.copy()
            self.search_group_ids = self.group_ids

        self.search_multimodal_X = {}
        for key, columns in self.multimodal_key.items():
            self.search_multimodal_X[key] = self.search_X[columns]
            self.multimodal_X = self.X[columns]

        for img_key in image:
            dataset["hash_" + img_key] = np.array(
                [np.asarray(img).sum() for img in dataset[img_key].to_numpy()]
            )
        self.internal_name = dataframe_hash(dataset[dataset.columns.difference(image)])
        for img_key in image:
            dataset.drop("hash_" + img_key, axis=1)

        self.study_name = study_name if study_name is not None else self.internal_name

        self.output_folder = Path(workspace) / self.study_name
        self.output_folder.mkdir(parents=True, exist_ok=True)

        self.output_file = self.output_folder / "model.p"

        self.num_study_iter = num_study_iter

        self.metric = metric
        self.score_threshold = score_threshold
        self.random_state = random_state
        self.n_folds_cv = n_folds_cv

        self.seeker = MultimodalEnsembleSeeker(
            self.internal_name,
            num_iter=num_iter,
            num_ensemble_iter=num_ensemble_iter,
            timeout=timeout,
            metric=metric,
            multimodal_type=multimodal_type,
            feature_scaling=feature_scaling,
            feature_selection=feature_selection,
            preprocess_images=preprocess_images,
            image_processing=image_processing,
            image_dimensionality_reduction=image_dimensionality_reduction,
            imputers=imputers,
            fusion=fusion,
            classifiers=classifiers,
            image_classifiers=image_classifiers,
            hooks=self.hooks,
            random_state=self.random_state,
            ensemble_size=ensemble_size,
            n_folds_cv=n_folds_cv,
            multimodal_key=self.multimodal_key,
        )

    def _should_continue(self) -> None:
        if self.hooks.cancel():
            raise StudyCancelled("Classifier study search cancelled")

    def _load_progress(self) -> Tuple[int, Any]:
        self._should_continue()

        if not self.output_file.is_file():
            return -1, None

        try:
            start = time.time()
            best_model = load_model_from_file(self.output_file)
            metrics = evaluate_multimodal_estimator(
                best_model,
                self.search_multimodal_X,
                self.search_Y,
                self.multimodal_type,
                metric=self.metric,
                group_ids=self.search_group_ids,
                n_folds=self.n_folds_cv,
            )
            best_score = metrics["raw"][self.metric][0]
            eval_metrics = {}
            for metric in metrics["raw"]:
                eval_metrics[metric] = metrics["raw"][metric][0]
                eval_metrics[f"{metric}_str"] = metrics["str"][metric]

            self.hooks.heartbeat(
                topic="classification_study",
                subtopic="candidate",
                event_type="candidate",
                name=best_model.name(),
                duration=time.time() - start,
                score=best_score,
                **eval_metrics,
            )

            return best_score, best_model
        except BaseException:
            return -1, None

    def _save_progress(self, model: Any) -> None:
        self._should_continue()

        if self.output_file:
            save_model_to_file(self.output_file, model)

    def run(self) -> Any:
        """Run the study. The call returns the optimal model architecture - not fitted."""
        self._should_continue()

        log.info("Start running study")

        best_score, best_model = self._load_progress()
        score = best_score

        patience = 0
        for it in range(self.num_study_iter):
            self._should_continue()
            start = time.time()

            current_model = self.seeker.search(
                self.search_multimodal_X, self.search_Y, group_ids=self.search_group_ids
            )

            metrics = evaluate_multimodal_estimator(
                current_model,
                self.search_multimodal_X,
                self.search_Y,
                self.multimodal_type,
                metric=self.metric,
                group_ids=self.search_group_ids,
                n_folds=self.n_folds_cv,
            )
            score = metrics["raw"][self.metric][0]
            eval_metrics = {}
            for metric in metrics["raw"]:
                eval_metrics[metric] = metrics["raw"][metric][0]
                eval_metrics[f"{metric}_str"] = metrics["str"][metric]

            self.hooks.heartbeat(
                topic="classification_study",
                subtopic="candidate",
                event_type="candidate",
                name=current_model.name(),
                duration=time.time() - start,
                score=score,
                **eval_metrics,
            )

            if score < self.score_threshold:
                log.critical(
                    f"The ensemble is not good enough, keep searching {metrics['str']}"
                )
                continue

            if best_score >= score:
                log.info(
                    f"Model score not improved {score}. Previous best {best_score}"
                )
                patience += 1

                if patience > PATIENCE:
                    log.info(
                        f"Study not improved for {PATIENCE} iterations. Stopping..."
                    )
                    break
                continue

            patience = 0
            best_score = metrics["raw"][self.metric][0]
            best_model = current_model

            log.error(
                f"Best ensemble so far: {best_model.name()} with score {metrics['raw'][self.metric]}"
            )

            self._save_progress(best_model)

        self.hooks.finish()

        if best_score < self.score_threshold:
            log.critical(
                f"Unable to find a model above threshold {self.score_threshold}. Returning None"
            )
            return None

        return best_model

    def fit(self) -> Any:
        """Run the study and train the model. The call returns the fitted model."""
        model = self.run()
        model.fit(self.multimodal_X, self.Y)

        return model
