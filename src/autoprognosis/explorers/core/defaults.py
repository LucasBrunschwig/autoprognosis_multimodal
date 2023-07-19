# third party - pytorch predefined architecture
# third party
from torchvision.models import (
    alexnet,
    densenet121,
    mobilenet_v3_large,
    resnet34,
    resnet50,
    vgg16,
    vgg16_bn,
)

# autoprognosis absolute
from autoprognosis.plugins.preprocessors import Preprocessors

default_classifiers_names = [
    "random_forest",
    "xgboost",
    "catboost",
    "lgbm",
    "logistic_regression",
]

default_image_classsifiers_names = ["cnn", "cnn_fine_tune"]

default_multimodal_names = ["intermediate_conv_net"]

default_regressors_names = [
    "random_forest_regressor",
    "xgboost_regressor",
    "linear_regression",
    "catboost_regressor",
]

default_imputers_names = ["mean", "ice", "missforest", "hyperimpute"]
default_image_processing = ["resizer", "normalizer"]
default_image_dimensionality_reduction = [
    "cnn",
    "cnn_fine_tune",
    "cnn_imagenet",
    "simsiam" "pca_image",
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
    "resnet50",
    "vgg16",
    "vgg16_bn",
    "mobilenet_v3_large",
    "densenet121",
]

CNN_MODEL = {
    "alexnet": alexnet,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "vgg16": vgg16,
    "vgg16_bn": vgg16_bn,
    "mobilenet_v3_large": mobilenet_v3_large,
    "densenet121": densenet121,
}

WEIGHTS = {
    "alexnet": "AlexNet_Weights.DEFAULT",
    "resnet34": "ResNet34_Weights.DEFAULT",
    "resnet50": "ResNet50_Weights.DEFAULT",
    "mobilenet_v3_large": "MobileNet_V3_Large_Weights.DEFAULT",
    "densenet121": "DenseNet121_Weights.DEFAULT",
    "vgg16": "VGG16_Weights.DEFAULT",
    "vgg16_bn": "VGG16_BN_Weights.DEFAULT",
    "vgg19": "VGG19_Weights.DEFAULT",
}

percentile_val = 1.96
