"""Script to manage config files, reduce clutter, and aid in iterative experiments."""

import argparse
import os
import pprint
import yaml

from typing import Literal
from matdeeplearn.common.config.build_config import build_config
