# autoprognosis absolute
from autoprognosis.plugins.preprocessors import Preprocessors

default_classifiers_names = [
    "random_forest",
    "xgboost",
    "catboost",
    "lgbm",
    "logistic_regression",
]

default_image_classsifiers_names = ["cnn"]

default_multimodal_names = [
    "early_fusion",
]
default_conv_models = [
    "AlexNet",
    "ResNet",
    "InceptionNet",
    "ZfNet",
    "VGG",
    "MobileNet",
]

default_regressors_names = [
    "random_forest_regressor",
    "xgboost_regressor",
    "linear_regression",
    "catboost_regressor",
]

default_imputers_names = ["mean", "ice", "missforest", "hyperimpute"]
default_image_processing = ["normalizer"]
default_image_dimensionality_reduction = [
    "pca_image",
    "fast_ica_image",
    "predefined_cnn",
]
default_modalities = ["img"]

default_feature_scaling_names = Preprocessors(
    category="feature_scaling"
).list_available()
default_feature_selection_names = ["nop", "pca", "fast_ica"]
default_risk_estimation_names = [
    "survival_xgboost",
    "loglogistic_aft",
    "deephit",
    "cox_ph",
    "weibull_aft",
    "lognormal_aft",
    "coxnet",
]

CNN = ["alexnet", "resnet18", "resnet50", "vgg19"]

WEIGHTS = {
    "alexnet": "AlexNet_Weights.DEFAULT",
    "resnet18": "ResNet18_Weights.DEFAULT",
    "resnet50": "ResNet50_Weights.DEFAULT",
    "vgg19": "VGG19_Weights.DEFAULT",
}


percentile_val = 1.96
