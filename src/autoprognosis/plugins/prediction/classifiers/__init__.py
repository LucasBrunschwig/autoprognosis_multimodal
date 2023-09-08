# stdlib
import glob
from os.path import basename, dirname, isfile, join

# autoprognosis absolute
from autoprognosis.plugins.core.base_plugin import PluginLoader
from autoprognosis.plugins.prediction.classifiers.base import (  # noqa: F401,E402
    ClassifierPlugin,
)

tabular_plugins = glob.glob(join(dirname(__file__), "tabular/plugin*.py"))
image_plugins = glob.glob(join(dirname(__file__), "image/plugin*.py"))
intermediate_plugins = glob.glob(join(dirname(__file__), "multimodal/plugin*.py"))


class Classifiers(PluginLoader):
    def __init__(self, category="all") -> None:
        # TMP Lucas:
        if category == "tabular":
            plugins = tabular_plugins
        elif category == "image":
            plugins = image_plugins
        elif category == "multimodal":
            plugins = intermediate_plugins
        elif category == "all":
            plugins = tabular_plugins + image_plugins + intermediate_plugins

        super().__init__(plugins, ClassifierPlugin)


__all__ = (
    [basename(f)[:-3] for f in tabular_plugins if isfile(f)]
    + [basename(f)[:-3] for f in image_plugins if isfile(f)]
    + [basename(f)[:-3] for f in intermediate_plugins if isfile(f)]
    + [
        "Classifiers",
        "ClassifierPlugin",
    ]
)
