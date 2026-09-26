"""Compatibility path; the six-method implementation lives in rdgdpo/."""
import sys
from rdgdpo import verl_adapter as _implementation
sys.modules[__name__] = _implementation
