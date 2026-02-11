"""
Compatibility patch for basicsr with newer torchvision versions.
This creates the missing functional_tensor module that basicsr expects.
"""

# Import from the actual location (with underscore prefix)
from torchvision.transforms._functional_tensor import *

# Re-export everything so basicsr can import it
__all__ = [
    'rgb_to_grayscale',
    # Add other functions as needed
]
