# Model options
# Use this file to control model settings and options
# Change settings in Main_New.py prior to running

from dataclasses import dataclass
from enum import Enum

class FeSpecKey(str, Enum):
    KRESS = "Kress"
    SOSSI = "Sossi"

@dataclass
class ModelOptions:
    escape_on: bool = True
    #reductants_on: bool = True
    #graphite_on: bool = True
    #ironsink_mantle = True
    #fe_speciation: FeSpecKey   = "Kress"

MODEL_OPTS = ModelOptions()


'''# Define switches to control model run settings. Change these settings in Main_New.py for each run
class ModelOptions:
    def __init__(self,
                 escape_on = True,
                 reductants_on = True,
                 graphite_on = True,
                 ironsink_mantle = True):
        self.escape_on = escape_on
        self.reductants_on = reductants_on
        self.graphite_on = graphite_on
        self.ironsink_mantle = ironsink_mantle

MODEL_OPTS = ModelOptions(  # automatice configuration for full physics
    escape_on=True,         # allows atmospheric escape
    reductants_on=True,     # allows reduced atmosphere
    graphite_on=True,       # allows graphite formation
    ironsink_mantle=True    # allows metallic iron to stay in mantle (false-> metallic iron immediately sequentered to core)
)'''