

from absl.flags import FLAGS
from settings import dataset
if dataset == "default":
    from core.config_default import *
if dataset == "wa":
    from core.config_wa import *
