# stdlib
from pathlib import Path
import time
from typing import Any, List, Optional, Tuple

# third party
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# autoprognosis absolute
from autoprognosis.exceptions import StudyCancelled
from autoprognosis.explorers.core.defaults import (
    default_image_classsifiers_names,
    default_image_processing,
)
from autoprognosis.explorers.core.selector import predefined_args
from autoprognosis.explorers.image_classifiers_combos import ImageEnsembleSeeker
from autoprognosis.hooks import DefaultHooks, Hooks
import autoprognosis.logger as log
from autoprognosis.studies._base import Study
from autoprognosis.utils.distributions import enable_reproducible_results
from autoprognosis.utils.serialization import (
    dataframe_hash,
    load_model_from_file,
    save_model_to_file,
)
from autoprognosis.utils.tester import evaluate_estimator

PATIENCE = 10
SCORE_THRESHOLD = 0.65


class ImageClassifierStudy(Study):
    """
    Core logic for image classification

    A study automatically handles imputation, preprocessing and model selection for a certain dataset.
    The output is an optimal model architecture, selected by the AutoML logic.

    Args:
        dataset: DataFrame.
            The dataset to analyze.
        target: str.
            The target column in the dataset.
        preprocess_images: bool.
            Specify if you require image preprocessing optimization
        predefined_cnn: list[str].
            Optional: specify which predefined architectures the pipeline will optimize.
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
                - "balanced_accuracy" : Accuracy classification balancing with class imbalance
                - "f1_score_micro": F1 score is a harmonic mean of the precision and recall. This version uses the "micro" average: calculate metrics globally by counting the total true positives, false negatives and false positives.
                - "f1_score_macro": F1 score is a harmonic mean of the precision and recall. This version uses the "macro" average: calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
                - "f1_score_weighted": F1 score is a harmonic mean of the precision and recall. This version uses the "weighted" average: Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).
                - "mcc": The Matthews correlation coefficient is used in machine learning as a measure of the quality of binary and multiclass classifications. It takes into account true and false positives and negatives and is generally regarded as a balanced measure which can be used even if the classes are of very different sizes.
                - "kappa", "kappa_quadratic":  computes Cohen’s kappa, a score that expresses the level of agreement between two annotators on a classification problem.
        study_name: str.
            The name of the study, to be used in the caches.
        image_processing: list.
            Plugin search pipeline to use in the pipeline for optimal image preprocessing. If you don't require image
            preprocessing you can specify preprocessing = False in the argument
            Available retrieved using `Preprocessors(category="image_processing").list_available()`
                1. 'resizer'
                2. 'normalizer'
        classifiers: list.
            Plugin search pool to use in the pipeline for prediction. Defaults to ["cnn"].
                - 'cnn'
                - 'cnn_fine_tune
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
        TODO: Change that
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
        >>> study = ClassifierStudy(
        >>>     study_name=study_name,
        >>>     dataset=df,  # pandas DataFrame
        >>>     target="target",  # the label column in the dataset
        >>> )
        >>> model = study.fit()
        >>>
        >>> # Predict the probabilities of each class using the model
        >>> model.predict_proba(X)
    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        target: str,
        preprocess_images: bool = True,
        predefined_cnn: list = [],
        num_iter: int = 20,
        num_study_iter: int = 5,
        num_ensemble_iter: int = 15,
        timeout: int = 3600,
        metric: str = "aucroc",
        study_name: Optional[str] = None,
        image_processing: List[str] = default_image_processing,
        classifiers: List[str] = default_image_classsifiers_names,
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

        if dataset[dataset.columns[0]].isna().sum() > 0:
            raise RuntimeError("This framework does not allow for missing images")

        if nan_placeholder is not None:
            dataset = dataset.replace(nan_placeholder, np.nan)

        if dataset.isnull().values.any():
            raise RuntimeError("Image classifiers does not handle missing features")

        if not preprocess_images:
            image_processing = []

        if predefined_cnn:
            if not isinstance(predefined_cnn, list):
                predefined_cnn = [predefined_cnn]
            predefined_args["predefined_cnn"] = predefined_cnn

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

        dataset["hash_image"] = np.array(
            [np.asarray(img).sum() for img in dataset["image"].to_numpy()]
        )
        self.internal_name = dataframe_hash(
            dataset[dataset.columns.difference(["image"])]
        )
        dataset.drop(columns=["hash_image"], inplace=True)

        self.study_name = study_name if study_name is not None else self.internal_name

        self.output_folder = Path(workspace) / self.study_name
        self.output_folder.mkdir(parents=True, exist_ok=True)

        self.output_file = self.output_folder / "model.p"

        self.num_study_iter = num_study_iter

        self.metric = metric
        self.score_threshold = score_threshold
        self.random_state = random_state
        self.n_folds_cv = n_folds_cv

        self.seeker = ImageEnsembleSeeker(
            self.internal_name,
            num_iter=num_iter,
            num_ensemble_iter=num_ensemble_iter,
            preprocess_images=preprocess_images,
            timeout=timeout,
            metric=metric,
            image_processing=image_processing,
            classifiers=classifiers,
            hooks=self.hooks,
            random_state=self.random_state,
            ensemble_size=ensemble_size,
            n_folds_cv=n_folds_cv,
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
            metrics = evaluate_estimator(
                best_model,
                self.search_X,
                self.search_Y,
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

        best_score, best_model = self._load_progress()
        score = best_score

        patience = 0
        for it in range(self.num_study_iter):
            self._should_continue()
            start = time.time()

            current_model = self.seeker.search(
                self.search_X, self.search_Y, group_ids=self.search_group_ids
            )

            metrics = evaluate_estimator(
                current_model,
                self.search_X,
                self.search_Y,
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

            log.info(
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

        self.Y = LabelEncoder().fit_transform(self.Y)
        self.Y = pd.Series(self.Y).reset_index(drop=True)

        model.fit(self.X, self.Y)

        return model
