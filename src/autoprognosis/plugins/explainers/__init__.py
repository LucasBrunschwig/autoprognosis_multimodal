# stdlib
import glob
from os.path import basename, dirname, isfile, join

# autoprognosis absolute
from autoprognosis.plugins.core.base_plugin import PluginLoader

# autoprognosis relative
from .base import ExplainerPlugin  # noqa: F401,E402

plugins_tabular = glob.glob(join(dirname(__file__), "tabular/plugin*.py"))
plugins_image = glob.glob(join(dirname(__file__), "image/plugin*.py"))
plugins_multimodal = glob.glob(join(dirname(__file__), "multimodal/plugin*.py"))

plugins_all = plugins_tabular + plugins_image + plugins_multimodal


class Explainers(PluginLoader):
    def __init__(self, data_type=None) -> None:

        if data_type == "multimodal":
            plugins = plugins_multimodal
        elif data_type == "image":
            plugins = plugins_image
        elif data_type == "tabular":
            plugins = plugins_tabular
        else:
            plugins = plugins_all

        super().__init__(plugins, ExplainerPlugin)

    def explainer_type(self, name):
        path_ = self.list_available_path()
        if path_.get(name, None):
            if "\\" in path_:
                return path_[name].split("\\")[-2]
            elif "/" in path_:
                return path_[name].split("/")[-2]

        else:
            raise RuntimeError(f"Plugin {name} does not exist")


__all__ = [basename(f)[:-3] for f in plugins_all if isfile(f)] + [
    "Explainers",
    "ExplainerPlugin",
]
