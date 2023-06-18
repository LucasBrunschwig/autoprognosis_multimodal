# autoprognosis absolute
from autoprognosis.plugins.preprocessors import Preprocessors

default_classifiers_names = [
    "random_forest",
    "xgboost",
    "catboost",
    "lgbm",
    "logistic_regression",
]

default_image_classsifiers_names = ["cnn_fine_tune", "cnn"]

default_multimodal_names = ["intermediate_conv_net", "intermediate_neural_net"]

default_regressors_names = [
    "random_forest_regressor",
    "xgboost_regressor",
    "linear_regression",
    "catboost_regressor",
]

default_imputers_names = ["mean", "ice", "missforest", "hyperimpute"]
default_image_processing = ["normalizer"]
default_image_dimensionality_reduction = [
    "predefined_cnn",
    "pca_image",
]
default_fusion = ["concatenate"]

IMAGE_KEY = "img"
TABULAR_KEY = "tab"
MULTIMODAL_KEY = "multimodal"

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

CNN = [
    "alexnet",
    "resnet34",
    #    "vgg16",
    #    "vgg19",
    #   "mobilenet_v3_large",
    #   "densenet121",
]

WEIGHTS = {
    "alexnet": "AlexNet_Weights.DEFAULT",
    "resnet18": "ResNet18_Weights.DEFAULT",
    "resnet34": "ResNet34_Weights.DEFAULT",
    "mobilenet_v3_large": "MobileNet_V3_Large_Weights.DEFAULT",
    "densenet121": "DenseNet121_Weights.DEFAULT",
    #    "vgg16": "VGG16_Weights.DEFAULT",
    #    "vgg19": "VGG19_Weights.DEFAULT",
}


percentile_val = 1.96
