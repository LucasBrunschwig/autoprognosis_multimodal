# stdlib
import glob
from os.path import basename, dirname, isfile, join

# autoprognosis absolute
from autoprognosis.plugins.core.base_plugin import PluginLoader

# autoprognosis relative
from .base import PreprocessorPlugin  # noqa: F401,E402

feature_scaling_plugins = glob.glob(
    join(dirname(__file__), "feature_scaling/plugin*.py")
)
dim_reduction_plugins = glob.glob(
    join(dirname(__file__), "dimensionality_reduction/plugin*.py")
)
image_processing_plugins = glob.glob(
    join(dirname(__file__), "image_processing/plugin*.py")
)
image_dimensionality_reduction_plugins = glob.glob(
    join(dirname(__file__), "image_reduction/plugin*.py")
)

multimodal_fusion_plugins = glob.glob(join(dirname(__file__), "fusion/plugin*.py"))


class Preprocessors(PluginLoader):
    def __init__(self, category: str = "feature_scaling") -> None:
        if category not in [
            "feature_scaling",
            "dimensionality_reduction",
            "image_processing",
            "image_reduction",
            "fusion",
        ]:
            raise RuntimeError("Invalid preprocessing category")

        self.category = category
        if category == "feature_scaling":
            plugins = feature_scaling_plugins
        elif category == "dimensionality_reduction":
            plugins = dim_reduction_plugins
        elif category == "image_processing":
            plugins = image_processing_plugins
        elif category == "image_reduction":
            plugins = image_dimensionality_reduction_plugins
        elif category == "fusion":
            plugins = multimodal_fusion_plugins

        super().__init__(plugins, PreprocessorPlugin)


__all__ = (
    [basename(f)[:-3] for f in feature_scaling_plugins if isfile(f)]
    + [basename(f)[:-3] for f in dim_reduction_plugins if isfile(f)]
    + [basename(f)[:-3] for f in image_processing_plugins if isfile(f)]
    + [basename(f)[:-3] for f in image_dimensionality_reduction_plugins if isfile(f)]
    + [basename(f)[:-3] for f in multimodal_fusion_plugins if isfile(f)]
    + [
        "Preprocessors",
        "PreprocessorPlugin",
    ]
)
