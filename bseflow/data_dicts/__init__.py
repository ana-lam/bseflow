from .model_variations import model_variations
from .drake_factors import drake_factors
import json
import os

with open(os.path.join(os.path.dirname(__file__), 'formationChannels.json')) as f:
    formation_channels = json.load(f)