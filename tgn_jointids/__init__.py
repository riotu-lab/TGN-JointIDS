"""TGN-JointIDS: long-tail-aware streaming IoT intrusion detection via
temporal graph memory and multi-task learning."""

__version__ = "1.0.0"

from .models.memory import TGNNodeMemory
from .models.tgn_jointids import TGNJointIDS

__all__ = ["TGNNodeMemory", "TGNJointIDS", "__version__"]
