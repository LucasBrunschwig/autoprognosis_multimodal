# stdlib
import copy
from typing import List, Optional, Tuple

# third party
import numpy as np
import pandas as pd
from pydantic import validate_arguments
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

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
)
from autoprognosis.explorers.core.optimizer import EnsembleOptimizer
from autoprognosis.explorers.core.selector import PipelineSelector
from autoprognosis.hooks import DefaultHooks, Hooks
import autoprognosis.logger as log
from autoprognosis.plugins.ensemble.classifiers import (
    AggregatingEnsemble,
    BaseEnsemble,
    WeightedEnsemble,
)
from autoprognosis.utils.default_modalities import IMAGE_KEY, TABULAR_KEY
from autoprognosis.utils.tester import evaluate_multimodal_estimator

# autoprognosis relative
from .classifiers import ClassifierSeeker
from .image_classifiers import ImageClassifierSeeker
from .multimodal_classifiers import MultimodalClassifierSeeker

EPS = 1e-8


class MultimodalEnsembleSeeker:
    """
    AutoML core logic for classification ensemble search in multimodal settings.

    Args:
        study_name: str.
            Study ID, used for caching keys.
        num_iter: int.
            Maximum Number of optimization trials. This is the limit of trials for each base estimator in the "classifiers" list, used in combination with the "timeout" parameter. For each estimator, the search will end after "num_iter" trials or "timeout" seconds.
        num_ensemble_iter: int.
            Number of optimization trials for the ensemble weights.
        timeout: int.
            Maximum wait time(seconds) for each estimator hyperparameter search. This timeout will apply to each estimator in the "classifiers" list.
        n_folds_cv: int.
            Number of folds to use for evaluation
        ensemble_size: int.
            Number of base models for the ensemble.
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
                - "kappa":  computes Cohen’s kappa, a score that expresses the level of agreement between two annotators on a classification problem.
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
        image_processing: list.
            Plugin search pipeline to use in the pipeline for optimal preprocessing. If the list is empty, the program
            assumes that you preprocessed the images yourself.
            Available retrieved using `Preprocessors(category="image_processing").list_available()`
                - 'normalizer'
                - 'resizer'
        image_dimensionality_reduction: list.
            Plugin search pool to use in the pipeline for optimal dimensionality reduction.
            Available retrieved using `Preprocessors(category="image_reduction").list_available()`
                - 'fast_ica_image'
                - 'pca_image'
                - 'predefined_cnn'
        fusion: list.
            Plugin search pool to use in the pipeline for optimal early modality fusion.
            Available retrieved using `Preprocessors(category="fusion").list_available()`
                - 'concatenate'
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
        image_classifiers: list.
            Plugin search pool to use in the pipeline for prediction. Defaults to ["cnn"].
                - 'cnn'
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
        random_state: int:
            Random seed
    """

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        study_name: str,
        num_iter: int = 100,
        num_ensemble_iter: int = 100,
        timeout: Optional[int] = 360,
        n_folds_cv: int = 5,
        ensemble_size: int = 3,
        metric: str = "aucroc",
        multimodal_type: str = "early_fusion",
        feature_scaling: List[str] = default_feature_scaling_names,
        feature_selection: List[str] = default_feature_selection_names,
        preprocess_images: bool = True,
        image_processing: List[str] = default_image_processing,
        image_dimensionality_reduction: List[
            str
        ] = default_image_dimensionality_reduction,
        fusion: List[str] = default_fusion,
        classifiers: List[str] = default_classifiers_names,
        image_classifiers: List[str] = default_image_classsifiers_names,
        imputers: List[str] = [],
        hooks: Hooks = DefaultHooks(),
        optimizer_type: str = "bayesian",
        random_state: int = 0,
        multimodal_key: dict = None,
    ) -> None:
        ensemble_size = min(ensemble_size, len(classifiers))

        self.num_iter = num_ensemble_iter
        self.timeout = timeout
        self.ensemble_size = ensemble_size
        self.n_folds_cv = n_folds_cv
        self.metric = metric
        self.study_name = study_name
        self.hooks = hooks
        self.optimizer_type = optimizer_type
        self.random_state = random_state
        self.multimodal_type = multimodal_type
        self.multimodal_key = multimodal_key

        # In case of early/intermediate fusion use both modality type
        if self.multimodal_type in ["early_fusion", "intermediate_fusion"]:
            self.seeker = MultimodalClassifierSeeker(
                study_name,
                num_iter=num_iter,
                metric=metric,
                n_folds_cv=n_folds_cv,
                top_k=ensemble_size,
                timeout=timeout,
                feature_scaling=feature_scaling,
                feature_selection=feature_selection,
                image_processing=image_processing,
                image_dimensionality_reduction=image_dimensionality_reduction,
                fusion=fusion,
                classifiers=classifiers,
                hooks=hooks,
                imputers=imputers,
                optimizer_type=optimizer_type,
                random_state=self.random_state,
                multimodal_key=self.multimodal_key,
                multimodal_type=self.multimodal_type,
            )

        # In case of late fusion one model per modality
        if self.multimodal_type == "late_fusion":
            self.image_seeker = ImageClassifierSeeker(
                study_name,
                num_iter=num_iter,
                metric=metric,
                n_folds_cv=n_folds_cv,
                top_k=ensemble_size,
                timeout=timeout,
                preprocess_images=preprocess_images,
                image_processing=image_processing,
                image_dimensionality_reduction=[],
                classifiers=image_classifiers,
                hooks=hooks,
                optimizer_type=optimizer_type,
                multimodal_type=multimodal_type,
                random_state=self.random_state,
            )
            self.tabular_seeker = ClassifierSeeker(
                study_name,
                num_iter=num_iter,
                metric=metric,
                n_folds_cv=n_folds_cv,
                top_k=ensemble_size,
                timeout=timeout,
                feature_scaling=feature_scaling,
                feature_selection=feature_selection,
                classifiers=classifiers,
                hooks=hooks,
                imputers=imputers,
                optimizer_type=optimizer_type,
                multimodal_type=self.multimodal_type,
                random_state=self.random_state,
            )

    def _should_continue(self) -> None:
        if self.hooks.cancel():
            raise StudyCancelled("Classifier combo search cancelled")

    def pretrain_for_cv(
        self,
        ensemble: list,
        X: dict,
        Y: pd.Series,
        group_ids: Optional[pd.Series] = None,
        seed: int = 0,
    ) -> List:
        self._should_continue()

        for X_mod, X_df in X.items():
            X[X_mod] = X_df.reset_index(drop=True)

        if group_ids is not None:
            skf = StratifiedGroupKFold(
                n_splits=self.n_folds_cv, shuffle=True, random_state=seed
            )
        else:
            skf = StratifiedKFold(
                n_splits=self.n_folds_cv, shuffle=True, random_state=seed
            )

        folds = []
        X_train = {}
        for train_index, _ in skf.split(np.zeros(Y.shape[0]), Y, groups=group_ids):

            for mod_ in X.keys():
                X_train[mod_] = X[mod_].loc[X[mod_].index[train_index]]
            Y_train = Y.loc[Y.index[train_index]]

            local_fold = []
            for estimator in ensemble:
                model = copy.deepcopy(estimator)
                model.fit(X_train, Y_train)
                local_fold.append(model)
            folds.append(local_fold)
        return folds

    def search_weights(
        self,
        ensemble: List,
        X: dict,
        Y: pd.Series,
        group_ids: Optional[pd.Series] = None,
    ) -> Tuple[WeightedEnsemble, float]:
        self._should_continue()

        Y = (
            pd.DataFrame(LabelEncoder().fit_transform(Y))
            .reset_index(drop=True)
            .squeeze()
        )

        pretrained_models = self.pretrain_for_cv(ensemble, X, Y, group_ids=group_ids)

        def evaluate(weights: List) -> float:
            self._should_continue()

            folds = []
            for fold in pretrained_models:
                folds.append(WeightedEnsemble(fold, weights))

            try:
                metrics = evaluate_multimodal_estimator(
                    folds,
                    X,
                    Y,
                    self.n_folds_cv,
                    pretrained=True,
                    group_ids=group_ids,
                )
            except BaseException as e:
                log.error(f"evaluate_ensemble failed: {e}")

                return 0

            log.debug(f"ensemble {folds[0].name()} : results {metrics['raw']}")
            score = metrics["raw"][self.metric][0]

            return score

        study = EnsembleOptimizer(
            study_name=f"{self.study_name}_classifier_exploration_ensemble_{self.metric}",
            ensemble_len=len(ensemble),
            evaluation_cbk=evaluate,
            optimizer_type=self.optimizer_type,
            n_trials=self.num_iter,
            timeout=self.timeout,
            random_state=self.random_state,
        )

        best_score, selected_weights = study.evaluate()
        weights = []
        for idx in range(len(ensemble)):
            weights.append(selected_weights[f"weight_{idx}"])

        weights = weights / (np.sum(weights) + EPS)
        log.info(f"Best trial for ensemble: {best_score} for {weights}")

        return WeightedEnsemble(ensemble, weights), best_score

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def search(
        self,
        X: dict,
        Y: pd.Series,
        group_ids: Optional[pd.Series] = None,
    ) -> BaseEnsemble:
        self._should_continue()

        if self.multimodal_type == "late_fusion":

            # Optimal Image Models
            predefined = True
            if not predefined:

                # Optimal Image Models
                best_image_models = self.image_seeker.search(
                    X[IMAGE_KEY], Y, group_ids=group_ids
                )

                # Optimal Tabular Models
                best_tabular_models = self.tabular_seeker.search(
                    X[TABULAR_KEY], Y, group_ids=group_ids
                )

                if not isinstance(best_image_models, list):
                    best_image_models = [best_image_models]
                if not isinstance(best_tabular_models, list):
                    best_tabular_models = [best_tabular_models]
                best_models = best_tabular_models + best_image_models

            else:
                neural_nets = PipelineSelector(
                    classifier="neural_nets",
                    imputers=["ice"],
                    feature_selection=default_feature_selection_names,
                    feature_scaling=default_feature_scaling_names,
                    multimodal_type="late_fusion",
                )

                # Random Forest Evolution
                tabular_model = neural_nets.get_pipeline_from_named_args(
                    **{
                        "prediction.classifier.neural_nets.feature_scaling_candidate.feature_normalizer_maxabs_scaler_minmax_scaler_nop_normal_transform_scaler_uniform_transform": "minmax_scaler",
                        "prediction.classifier.neural_nets.feature_selection_candidate.fast_ica_nop_pca": "nop",
                        "preprocessor.dimensionality_reduction.pca.n_components": 7,
                        "preprocessor.dimensionality_reduction.fast_ica.n_components": 10,
                        "prediction.classifier.neural_nets.n_layers_hidden": 1,
                        "prediction.classifier.neural_nets.n_units_hidden": 86,
                        "prediction.classifier.neural_nets.lr": 0.0001,
                        "prediction.classifier.neural_nets.weight_decay": 0.001,
                        "prediction.classifier.neural_nets.dropout": 0.1,
                        "prediction.classifier.neural_nets.clipping_value": 1,
                    }
                )

                cnn_fine_tune = PipelineSelector(
                    classifier="cnn_fine_tune", multimodal_type="late_fusion"
                )

                image_model = cnn_fine_tune.get_image_pipeline_from_named_args(
                    **{
                        "prediction.classifier.conv_net": "alexnet",
                        "prediction.classifier.n_additional_layers": 2,
                        "prediction.classifier.lr": 1,
                        "prediction.classifier.n_unfrozen_layer": 7,
                        "prediction.classifier.weighted_cross_entropy": False,
                        "prediction.classifier.data_augmentation": "trivial_augment",
                        "prediction.classifier.replace_classifier": False,
                        "prediction.classifier.clipping_value": 0,
                    }
                )

                best_models = [image_model, tabular_model]

            scores = []
            ensembles: list = []

            # try:
            #     stacking_ensemble = StackingEnsemble(
            #         best_models, meta_model=best_models[0], use_proba=True
            #     )
            #     stacking_ens_score = evaluate_multimodal_estimator(
            #         stacking_ensemble,
            #         X,
            #         Y,
            #         self.n_folds_cv,
            #         group_ids=group_ids,
            #     )["raw"][self.metric][0]
            #     log.info(
            #         f"Stacking ensemble: {stacking_ensemble.name()} --> {stacking_ens_score}"
            #     )
            #     scores.append(stacking_ens_score)
            #     ensembles.append(stacking_ensemble)
            # except BaseException as e:
            #     log.info(f"StackingEnsemble failed {e}")

            if self.hooks.cancel():
                raise StudyCancelled("Classifier search cancelled")

            try:
                aggr_ensemble = AggregatingEnsemble(best_models)
                aggr_ens_score = evaluate_multimodal_estimator(
                    aggr_ensemble,
                    X,
                    Y,
                    n_folds=self.n_folds_cv,
                    group_ids=group_ids,
                )["raw"][self.metric][0]
                log.info(
                    f"Aggregating ensemble: {aggr_ensemble.name()} --> {aggr_ens_score}"
                )
                scores.append(aggr_ens_score)
                ensembles.append(aggr_ensemble)
            except BaseException as e:
                log.info(f"AggregatingEnsemble failed {e}")

            if self.hooks.cancel():
                raise StudyCancelled("Classifier search cancelled")

            weighted_ensemble, weighted_ens_score = self.search_weights(
                best_models, X, Y, group_ids=group_ids
            )
            log.info(
                f"Weighted ensemble: {weighted_ensemble.name()} -> {weighted_ens_score}"
            )

            scores.append(weighted_ens_score)
            ensembles.append(weighted_ensemble)

            if self.hooks.cancel():
                raise StudyCancelled("Classifier search cancelled")

            return ensembles[np.argmax(scores)]

        # Early fusion optimization
        elif self.multimodal_type == "early_fusion":

            # Optimize the learned representation
            predefined = True
            if not predefined:
                self.seeker.lr_search(X[IMAGE_KEY], Y, group_ids=group_ids)
            else:
                self.seeker.best_representation["cnn_fine_tune.50"] = {
                    "output_size": 50,
                    "conv_net": "alexnet",
                    "lr": 3,
                    "n_additional_layers": 1,
                    "n_unfrozen_layer": 6,
                    "data_augmentation": "gaussian_noise",
                    "clipping_value": 1,
                    "replace_classifier": True,
                }
                self.seeker.best_representation["cnn_fine_tune.100"] = {
                    "output_size": 100,
                    "conv_net": "alexnet",
                    "lr": 1,
                    "n_additional_layers": 1,
                    "n_unfrozen_layer": 6,
                    "data_augmentation": "gaussian_noise",
                    "clipping_value": 1,
                    "replace_classifier": False,
                }
                self.seeker.best_representation["cnn_fine_tune.300"] = {
                    "output_size": 300,
                    "conv_net": "alexnet",
                    "lr": 1,
                    "n_additional_layers": 1,
                    "n_unfrozen_layer": 8,
                    "data_augmentation": "autoaugment_imagenet",
                    "clipping_value": 1,
                    "replace_classifier": False,
                }
            # Pretrain and predict learned representation and use (LR key) if this works
            self.seeker.pretrain_lr_for_early_fusion(X[IMAGE_KEY], Y, group_ids)
            # Train the classifier - provide X
            best_models = self.seeker.search(X, Y, group_ids=group_ids)
            scores = []
            ensembles = []
            if len(best_models) > 1:
                for model in best_models:
                    try:
                        results = evaluate_multimodal_estimator(
                            model,
                            X,
                            Y,
                            n_folds=self.n_folds_cv,
                            group_ids=group_ids,
                        )
                        model_score = results["raw"][self.metric][0]

                        log.info(f"Model: {model.name()} --> {model_score}")

                        for name, metric in results["str"].items():
                            log.info(f"{name} {metric}")

                        ensembles.append(model)
                        scores.append(model_score)
                    except Exception as e:
                        log.error(f"Could not be fitted: {model.name()} - {e}")

                if self.hooks.cancel():
                    raise StudyCancelled("Classifier search cancelled")

                weighted_ensemble, weighted_ens_score = self.search_weights(
                    best_models, X, Y, group_ids=group_ids
                )

                log.info(
                    f"Weighted ensemble: {weighted_ensemble.name()} -> {weighted_ens_score}"
                )

                scores.append(weighted_ens_score)
                ensembles.append(weighted_ensemble)

                return ensembles[np.argmax(scores)]

            else:
                return best_models[0]

        # Intermediate fusion optimization
        elif self.multimodal_type == "intermediate_fusion":
            best_models = self.seeker.search(X, Y, group_ids=group_ids)

            if self.hooks.cancel():
                raise StudyCancelled("Classifier search cancelled")

            return best_models[0]

        else:
            raise ValueError(
                f"This type of multimodal study does not exist: {self.multimodal_type}"
            )
