# stdlib
from typing import Any, Dict, List, Tuple, Type, Union

# third party
from optuna.trial import Trial

# autoprognosis absolute
import autoprognosis.logger as log
from autoprognosis.plugins.core.base_plugin import Plugin
import autoprognosis.plugins.core.params as params
from autoprognosis.plugins.imputers import Imputers
from autoprognosis.plugins.pipeline import Pipeline, PipelineMeta
from autoprognosis.plugins.prediction import Predictions
from autoprognosis.plugins.preprocessors import Preprocessors

predefined_args = {
    "features_count": 10,
    "predefined_cnn": [],
}

LR_SEARCH = False


class PipelineSelector:
    """AutoML wrapper for pipelines

    Args:
        classifier: str
            Last estimator of the pipeline, the final classifier.
        calibration: int
            Type of calibration to use. 0 - none, 1 - sigmoid, 2 - isotonic.
        imputers: list
            list of imputers to sample from.
        feature_scaling: list
            list of feature scaling transformers to sample from.
        feature_selection: list
            list of feature selection methods to sample from
        preprocess_images: bool
            if image preprocessing should be optimized
        image_processing: list
            list of step in image preprocessing ordered list
        image_dimensionality_reduction: list
            list of image dimensionality reduction to sample from
        fusion: list
            list of fusion to sample from
        classifier_category: str
            task type: "classifier" or "risk_estimation"
        multimodal_type: str,
            type of multimodal fusion, None if unimodal
        data_type: str,

    """

    def __init__(
        self,
        classifier: str,
        calibration: List[int] = [0, 1, 2],
        imputers: List[str] = [],
        feature_scaling: List[str] = [],
        feature_selection: List[str] = [],
        preprocess_images: bool = False,
        image_processing: List[str] = [],
        image_dimensionality_reduction: List[str] = [],
        fusion: List[str] = [],
        classifier_category: str = "classifier",  # "classifier", "risk_estimation", "regression"
        multimodal_type: str = None,
    ) -> None:
        self.calibration = calibration
        self.preprocess_image = preprocess_images
        self.multimodal_type = multimodal_type
        self.imputers = [Imputers().get_type(plugin) for plugin in imputers]
        self.feature_scaling = [
            Preprocessors(category="feature_scaling").get_type(plugin)
            for plugin in feature_scaling
        ]
        self.feature_selection = [
            Preprocessors(category="dimensionality_reduction").get_type(plugin)
            for plugin in feature_selection
        ]

        self.image_dimensionality_reduction = [
            Preprocessors(category="image_reduction").get_type(plugin)
            for plugin in image_dimensionality_reduction
        ]
        self.fusion = [
            Preprocessors(category="fusion").get_type(plugin) for plugin in fusion
        ]

        if classifier == "multinomial_naive_bayes" or classifier == "bagging":
            self.feature_scaling = [
                Preprocessors(category="feature_scaling").get_type("minmax_scaler")
            ]
            self.feature_selection = []

        # To Discuss
        # if self.preprocess_image:
        #     self.image_processing = [
        #         Preprocessors(category="image_processing").get_type(plugin)
        #         for plugin in image_processing
        #     ]
        self.image_processing = image_processing
        self.image_processing = []
        self.preprocess_image = False

        classifier_type = Predictions(category=classifier_category).model_type(
            classifier
        )
        if classifier_type in ["multimodal", "image"]:  # can not be early fusion
            self.image_dimensionality_reduction = []
            self.fusion = []

        self.classifier = Predictions(
            category=classifier_category,
        ).get_type(classifier)

    def _generate_dist_name(self, key: str, step: str = None) -> str:
        if key == "imputation_candidate":
            imputers_str = [imputer.name() for imputer in self.imputers]
            imputers_str.sort()
            return f"{self.classifier.fqdn()}.imputation_candidate.{'_'.join(imputers_str)}"
        elif key == "feature_scaling_candidate":
            fs_str = [fs.name() for fs in self.feature_scaling]
            fs_str.sort()
            return (
                f"{self.classifier.fqdn()}.feature_scaling_candidate.{'_'.join(fs_str)}"
            )
        elif key == "feature_selection_candidate":
            fs_str = [fs.name() for fs in self.feature_selection]
            fs_str.sort()
            return f"{self.classifier.fqdn()}.feature_selection_candidate.{'_'.join(fs_str)}"
        elif key == "image_processing_step":
            return f"{self.classifier.fqdn()}.image_processing_step.{step}"
        elif key == "image_reduction_candidate":
            fs_str = [fs.name() for fs in self.image_dimensionality_reduction]
            fs_str.sort()
            return (
                f"{self.classifier.fqdn()}.image_reduction_candidate.{'_'.join(fs_str)}"
            )
        elif key == "fusion_candidate":
            fs_str = [fs.name() for fs in self.fusion]
            fs_str.sort()
            return f"{self.classifier.fqdn()}.fusion_candidate.{'_'.join(fs_str)}"

        else:
            raise ValueError(f"invalid key {key}")

    def hyperparameter_space(self) -> List:
        hp: List[Union[params.Integer, params.Categorical, params.Float]] = []

        if len(self.imputers) > 0:
            hp.append(
                params.Categorical(
                    self._generate_dist_name("imputation_candidate"),
                    [imputer.name() for imputer in self.imputers],
                )
            )
            for plugin in self.imputers:
                hp.extend(plugin.hyperparameter_space_fqdn(**predefined_args))

        if len(self.feature_scaling) > 0:
            hp.append(
                params.Categorical(
                    self._generate_dist_name("feature_scaling_candidate"),
                    [fs.name() for fs in self.feature_scaling],
                )
            )

        if len(self.feature_selection) > 0:
            hp.append(
                params.Categorical(
                    self._generate_dist_name("feature_selection_candidate"),
                    [fs.name() for fs in self.feature_selection],
                )
            )
            for plugin in self.feature_selection:
                hp.extend(plugin.hyperparameter_space_fqdn(**predefined_args))

        if len(self.image_processing) > 0:
            for fs in self.image_processing:
                hp.append(
                    params.Categorical(
                        self._generate_dist_name("image_processing_step", fs.name()),
                        [fs.name()],
                    )
                )
                hp.extend(fs.hyperparameter_space_fqdn(**predefined_args))

        if len(self.image_dimensionality_reduction) > 0:
            hp.append(
                params.Categorical(
                    self._generate_dist_name("image_reduction_candidate"),
                    [fs.name() for fs in self.image_dimensionality_reduction],
                )
            )
            for plugin in self.image_dimensionality_reduction:
                hp.extend(plugin.hyperparameter_space_fqdn(**predefined_args))
        if len(self.fusion) > 0:
            hp.append(
                params.Categorical(
                    self._generate_dist_name("fusion_candidate"),
                    [fs.name() for fs in self.fusion],
                )
            )
            for plugin in self.fusion:
                hp.extend(plugin.hyperparameter_space_fqdn(**predefined_args))

        if len(self.calibration) > 0:
            hp.append(
                params.Integer(
                    self.classifier.fqdn() + ".calibration",
                    0,
                    len(self.calibration) - 1,
                ),
            )

        hp.extend(self.classifier.hyperparameter_space_fqdn(**predefined_args))
        return hp

    def sample_hyperparameters(self, trial: Trial) -> Dict:
        params = self.hyperparameter_space()

        result = {}
        for param in params:
            result[param.name] = param.sample(trial)

        return result

    def sample_hyperparameters_np(self) -> Dict:
        params = self.hyperparameter_space()

        result = {}
        for param in params:
            result[param.name] = param.sample_np()

        return result

    def name(self) -> str:
        return self.classifier.name()

    def get_pipeline_template(
        self, search_domains: List[params.Params], hyperparams: List
    ) -> Tuple[List, Dict]:
        domain_list = [search_domains[k].name for k in range(len(search_domains))]

        model_list = list()

        calibration = hyperparams[
            domain_list.index(self.classifier.fqdn() + ".calibration")
        ]

        args: Dict[str, Dict] = {
            self.classifier.name(): {
                "calibration": int(calibration),
            },
        }

        def add_stage_hp(plugin: Type[Plugin]) -> None:
            if plugin.name() not in args:
                args[plugin.name()] = {}

            for param in plugin.hyperparameter_space_fqdn(**predefined_args):
                param_val = hyperparams[domain_list.index(param.name)]
                param_val = type(param.bounds[0])(param_val)

                args[plugin.name()][param.name.split(".")[-1]] = param_val

        if len(self.imputers) > 0:
            select_imp = hyperparams[
                domain_list.index(self._generate_dist_name("imputation_candidate"))
            ]
            selected = Imputers().get_type(select_imp)
            model_list.append(selected.fqdn())
            add_stage_hp(selected)

        if len(self.feature_scaling) > 0:
            select_pre_fs = hyperparams[
                domain_list.index(self._generate_dist_name("feature_scaling_candidate"))
            ]
            selected = Preprocessors(category="feature_scaling").get_type(select_pre_fs)
            model_list.append(selected.fqdn())
            add_stage_hp(selected)

        if len(self.feature_selection) > 0:
            select_pre_fs = hyperparams[
                domain_list.index(
                    self._generate_dist_name("feature_selection_candidate")
                )
            ]
            selected = Preprocessors(category="dimensionality_reduction").get_type(
                select_pre_fs
            )
            model_list.append(selected.fqdn())
            add_stage_hp(selected)

        # Add data cleanup
        cleaner = Preprocessors(category="dimensionality_reduction").get_type(
            "data_cleanup"
        )
        model_list.append(cleaner.fqdn())

        # Add predictor
        model_list.append(self.classifier.fqdn())
        add_stage_hp(self.classifier)

        log.info(f"[get_pipeline]: {model_list} -> {args}")

        return model_list, args

    def get_pipeline(
        self, search_domains: List[params.Params], hyperparams: List
    ) -> PipelineMeta:
        model_list, args = self.get_pipeline_template(search_domains, hyperparams)
        return self.get_pipeline_from_template(model_list, args)

    def get_pipeline_from_template(self, model_list: List, args: Dict) -> PipelineMeta:
        return Pipeline(model_list, self.multimodal_type)(args)

    def get_pipeline_from_named_args(self, **kwargs: Any) -> PipelineMeta:
        model_list = list()

        pipeline_args: dict = {}

        def add_stage_hp(plugin: Type[Plugin]) -> None:
            if plugin.name() not in pipeline_args:
                pipeline_args[plugin.name()] = {}

            for param in plugin.hyperparameter_space_fqdn(**predefined_args):
                if param.name not in kwargs:
                    continue
                param_val = kwargs[param.name]
                param_val = type(param.bounds[0])(param_val)

                pipeline_args[plugin.name()][param.name.split(".")[-1]] = param_val

        imputation_key = self._generate_dist_name("imputation_candidate")

        if imputation_key in kwargs:
            idx = kwargs[imputation_key]
            selected = Imputers().get_type(idx)
            model_list.append(selected.fqdn())
            add_stage_hp(selected)
        elif len(self.imputers) > 0:
            model_list.append(self.imputers[0].fqdn())
            add_stage_hp(self.imputers[0])

        pre_key = self._generate_dist_name("feature_selection_candidate")
        if pre_key in kwargs:
            idx = kwargs[pre_key]
            selected = Preprocessors(category="dimensionality_reduction").get_type(idx)
            model_list.append(selected.fqdn())
            add_stage_hp(selected)

        pre_key = self._generate_dist_name("feature_scaling_candidate")
        if pre_key in kwargs:
            idx = kwargs[pre_key]
            selected = Preprocessors(category="feature_scaling").get_type(idx)
            model_list.append(selected.fqdn())
            add_stage_hp(selected)

        # Add data cleanup
        cleaner = Preprocessors(category="dimensionality_reduction").get_type(
            "data_cleanup"
        )
        model_list.append(cleaner.fqdn())

        # Add predictor
        model_list.append(self.classifier.fqdn())
        add_stage_hp(self.classifier)

        return Pipeline(model_list, self.multimodal_type)(pipeline_args)

    def get_image_pipeline_from_named_args(self, **kwargs: Any) -> PipelineMeta:
        model_list = list()
        pipeline_args: dict = {}

        def add_stage_hp(plugin: Type[Plugin]) -> None:
            if plugin.name() not in pipeline_args:
                pipeline_args[plugin.name()] = {}

            for param in plugin.hyperparameter_space_fqdn(**predefined_args):
                if param.name not in kwargs:
                    continue
                param_val = kwargs[param.name]
                param_val = type(param.bounds[0])(param_val)

                pipeline_args[plugin.name()][param.name.split(".")[-1]] = param_val

        if self.preprocess_image:

            resizer = Preprocessors(category="image_processing").get_type("resizer")
            model_list.append(resizer.fqdn())
            add_stage_hp(resizer)
            normalizer = Preprocessors(category="image_processing").get_type(
                "normalizer"
            )
            model_list.append(normalizer.fqdn())
            add_stage_hp(normalizer)

        img_reduction_key = self._generate_dist_name("image_reduction_candidate")
        if img_reduction_key in kwargs:
            idx = kwargs[img_reduction_key]
            selected = Preprocessors(category="image_reduction").get_type(idx)
            model_list.append(selected.fqdn())
            add_stage_hp(selected)
        elif len(self.image_dimensionality_reduction) > 0:
            model_list.append(self.image_dimensionality_reduction[0].fqdn())
            add_stage_hp(self.image_dimensionality_reduction[0])

        # Add predictor
        model_list.append(self.classifier.fqdn())
        add_stage_hp(self.classifier)

        return Pipeline(model_list, self.multimodal_type)(pipeline_args)

    def get_multimodal_pipeline_from_named_args(self, **kwargs: Any) -> PipelineMeta:
        model_list = list()

        pipeline_args: dict = {}

        def add_stage_hp(plugin: Type[Plugin]) -> None:
            if plugin.name() not in pipeline_args:
                pipeline_args[plugin.name()] = {}

            for param in plugin.hyperparameter_space_fqdn(**predefined_args):
                if param.name not in kwargs:
                    continue
                param_val = kwargs[param.name]
                param_val = type(param.bounds[0])(param_val)

                pipeline_args[plugin.name()][param.name.split(".")[-1]] = param_val

        imputation_key = self._generate_dist_name("imputation_candidate")

        if imputation_key in kwargs:
            idx = kwargs[imputation_key]
            selected = Imputers().get_type(idx)
            model_list.append(selected.fqdn())
            add_stage_hp(selected)
        elif len(self.imputers) > 0:
            model_list.append(self.imputers[0].fqdn())
            add_stage_hp(self.imputers[0])

        pre_key = self._generate_dist_name("feature_selection_candidate")
        if pre_key in kwargs:
            idx = kwargs[pre_key]
            selected = Preprocessors(category="dimensionality_reduction").get_type(idx)
            model_list.append(selected.fqdn())
            add_stage_hp(selected)

        pre_key = self._generate_dist_name("feature_scaling_candidate")
        if pre_key in kwargs:
            idx = kwargs[pre_key]
            selected = Preprocessors(category="feature_scaling").get_type(idx)
            model_list.append(selected.fqdn())
            add_stage_hp(selected)

        # Add data cleanup
        cleaner = Preprocessors(category="dimensionality_reduction").get_type(
            "data_cleanup"
        )
        model_list.append(cleaner.fqdn())

        # Feature extraction through pretrained fine-tuning or imagenet pretrained network does not need preprocessing
        img_reduction_key = self._generate_dist_name("image_reduction_candidate")
        if self.preprocess_image and (
            (  # Check if the dimensionality reduction plugin uses predefined weights
                kwargs.get(img_reduction_key, None)
                and not kwargs.get(img_reduction_key)
                in ["cnn_fine_tune", "cnn_imagenet", "cnn"]
            )
            or (  # Check if the default dimensionality reduction plugin uses predefined weights
                not kwargs.get(img_reduction_key, None)
                and len(self.image_dimensionality_reduction) > 0
                and not (
                    self.image_dimensionality_reduction[0].fqdn().split(".")[-1]
                    in ["cnn_fine_tune", "cnn_imagenet", "cnn"]
                )
            )
        ):
            resizer = Preprocessors(category="image_processing").get_type("resizer")
            model_list.append(resizer.fqdn())
            add_stage_hp(resizer)
            normalizer = Preprocessors(category="image_processing").get_type(
                "normalizer"
            )
            model_list.append(normalizer.fqdn())
            add_stage_hp(normalizer)

        if img_reduction_key in kwargs:
            idx = kwargs[img_reduction_key]
            selected = Preprocessors(category="image_reduction").get_type(idx)
            model_list.append(selected.fqdn())
            add_stage_hp(selected)
        elif len(self.image_dimensionality_reduction) > 0:
            model_list.append(self.image_dimensionality_reduction[0].fqdn())
            add_stage_hp(self.image_dimensionality_reduction[0])

        fusion_key = self._generate_dist_name("fusion_candidate")
        if fusion_key in kwargs:
            idx = kwargs[fusion_key]
            selected = Preprocessors(category="fusion").get_type(idx)
            model_list.append(selected.fqdn())
            add_stage_hp(selected)
        elif len(self.fusion) > 0:
            model_list.append(self.fusion[0].fqdn())
            add_stage_hp(self.fusion[0])

        # Add predictor
        model_list.append(self.classifier.fqdn())
        add_stage_hp(self.classifier)

        return Pipeline(model_list, self.multimodal_type)(pipeline_args)

    def remove_tabular_processing(self):
        """This function removes the tabular processing steps if there are no tabular data"""
        self.imputers = []
        self.feature_selection = []
        self.feature_scaling = []

    def remove_image_processing(self):
        """This function removes the tabular processing steps if image are preprocessed"""
        self.image_processing = []
