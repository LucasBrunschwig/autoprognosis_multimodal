# stdlib
import time
from typing import Any, List, Optional, Tuple

# third party
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from pydantic import validate_arguments

# autoprognosis absolute
from autoprognosis.exceptions import StudyCancelled
from autoprognosis.explorers.core.defaults import (
    default_image_classsifiers_names,
    default_image_processing,
)
from autoprognosis.explorers.core.optimizer import Optimizer
from autoprognosis.explorers.core.selector import PipelineSelector
from autoprognosis.hooks import DefaultHooks, Hooks
import autoprognosis.logger as log
from autoprognosis.utils.parallel import n_opt_image_jobs
from autoprognosis.utils.tester import evaluate_estimator

dispatcher = Parallel(max_nbytes=None, backend="loky", n_jobs=n_opt_image_jobs())


class ImageClassifierSeeker:
    """
    AutoML core logic for classification tasks with biomedical images.

    Args:
        study_name: str.
            Study ID, used for caching.
        num_iter: int.
            Maximum Number of optimization trials. This is the limit of trials for each base estimator in the "classifiers" list, used in combination with the "timeout" parameter. For each estimator, the search will end after "num_iter" trials or "timeout" seconds.
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
        n_folds_cv: int.
            Number of folds to use for evaluation
        top_k: int
            Number of candidates to return
        timeout: int.
            Maximum wait time(seconds) for each estimator hyperparameter search. This timeout will apply to each estimator in the "classifiers" list.
        image_processing: list.
            Plugin search pipeline to use in the pipeline for optimal preprocessing. If the list is empty, the program
            assumes that you preprocessed the images yourself.
            Available retrieved using `Preprocessors(category="image_processing").list_available()`
                - 'normalizer'
                - 'resizer'
                - 'data_augmentation
        preprocess_images: bool,
            indicate if image require resizing
        classifiers: list.
            Plugin search pool to use in the pipeline for prediction. Defaults to ["cnn_fine_tune"]
            Available retrieved using 'Classifiers(category="image").list_available()'
                - 'cnn'
                - 'cnn_fine_tune'
                - 'vision_transformers'
        hooks: Hooks.
            Custom callbacks to be notified about the search progress.
        optimizer_type: str,
            choose the AutoML optimization algorithm between Bayesian or Hyper-Band
        strict: bool,
            stop the optimization if one process fails
        random_state: int:
            Random seed
    """

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        study_name: str,
        multimodal_type: str = None,
        num_iter: int = 100,
        metric: str = "aucroc",
        n_folds_cv: int = 5,
        top_k: int = 3,
        timeout: int = 360,
        image_processing: List[str] = default_image_processing,
        preprocess_images: bool = True,
        classifiers: List[str] = default_image_classsifiers_names,
        hooks: Hooks = DefaultHooks(),
        optimizer_type: str = "bayesian",
        strict: bool = False,
        random_state: int = 0,
    ) -> None:
        for int_val in [num_iter, n_folds_cv, top_k, timeout]:
            if int_val <= 0 or type(int_val) != int:
                raise ValueError(
                    f"invalid input number {int_val}. Should be a positive integer"
                )
        metrics = [
            "aucroc",
            "aucprc",
            "accuracy",
            "balanced_accuracy",
            "kappa",
            "f1_score_micro",
            "f1_score_macro",
            "f1_score_weighted",
            "mcc",
        ]
        if metric not in metrics:
            raise ValueError(f"invalid input metric. Should be from {metrics}")

        self.study_name = study_name
        self.hooks = hooks

        self.estimators = [
            PipelineSelector(
                plugin,
                calibration=[],
                preprocess_images=preprocess_images,
                image_dimensionality_reduction=[],
                image_processing=image_processing,
                multimodal_type=multimodal_type,
            )
            for plugin in classifiers
        ]

        self.n_folds_cv = n_folds_cv
        self.num_iter = num_iter
        self.strict = strict
        self.timeout = timeout
        self.top_k = top_k
        self.metric = metric
        self.optimizer_type = optimizer_type
        self.random_state = random_state

    def _should_continue(self) -> None:
        if self.hooks.cancel():
            raise StudyCancelled("Classifier search cancelled")

    def search_best_args_for_estimator(
        self,
        estimator: Any,
        X: pd.DataFrame,
        Y: pd.Series,
        group_ids: Optional[pd.Series] = None,
    ) -> Tuple[List[float], List[float]]:
        self._should_continue()

        def evaluate_args(**kwargs: Any) -> float:
            self._should_continue()

            start = time.time()

            model = estimator.get_image_pipeline_from_named_args(**kwargs)
            try:
                metrics = evaluate_estimator(
                    model, X, Y, n_folds=self.n_folds_cv, group_ids=group_ids
                )
            except BaseException as e:
                log.error(f"evaluate_estimator failed: {e}")

                if self.strict:
                    raise

                return 0

            eval_metrics = {}
            for metric in metrics["raw"]:
                eval_metrics[metric] = metrics["raw"][metric][0]
                eval_metrics[f"{metric}_str"] = metrics["str"][metric]

            self.hooks.heartbeat(
                topic="classification",
                subtopic="model_search",
                event_type="performance",
                name=model.name(),
                model_args=kwargs,
                duration=time.time() - start,
                score=metrics["str"][self.metric],
                **eval_metrics,
            )
            return metrics["raw"][self.metric][0]

        study = Optimizer(
            study_name=f"{self.study_name}_classifiers_exploration_{estimator.name()}_{self.metric}",
            estimator=estimator,
            evaluation_cbk=evaluate_args,
            optimizer_type=self.optimizer_type,
            n_trials=self.num_iter,
            timeout=self.timeout,
            random_state=self.random_state,
        )
        return study.evaluate()

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def search(
        self,
        X: pd.DataFrame,
        Y: pd.Series,
        group_ids: Optional[pd.Series] = None,
    ) -> List:
        """Search the optimal model for the task.

        Args:
            X: DataFrame
                The images, collection of addresses
            Y: DataFrame/Series
                The labels
            group_ids: Optional str
                Optional Group labels for the samples used while splitting the dataset into train/test set.
        """
        self._should_continue()

        search_results = dispatcher(
            delayed(self.search_best_args_for_estimator)(estimator, X, Y, group_ids)
            for estimator in self.estimators
        )

        all_scores = []
        all_args = []
        all_estimators = []

        for idx, (best_scores, best_args) in enumerate(search_results):
            best_idx = np.argmax(best_scores)
            all_scores.append(best_scores[best_idx])
            all_args.append(best_args[best_idx])
            all_estimators.append(self.estimators[idx])
            log.info(
                f"Evaluation for {self.estimators[idx].name()} scores: {max(best_scores)}."
            )

        all_scores_np = np.array(all_scores)
        selected_points = min(self.top_k, len(all_scores))
        best_scores = np.sort(np.unique(all_scores_np.ravel()))[-selected_points:]

        result = []
        for score in reversed(best_scores):
            pos = np.argwhere(all_scores_np == score)[0]
            pos_est = pos[0]
            log.info(
                f"Selected score {score}: {all_estimators[pos_est].name()} : {all_args[pos_est]}"
            )
            model = all_estimators[pos_est].get_image_pipeline_from_named_args(
                **all_args[pos_est]
            )
            result.append(model)

        return result
