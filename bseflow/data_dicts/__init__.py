from model_variations import model_variations

import json
import os

with open(os.path.join(os.path.dirname(__file__), 'drakeFactors.json')) as f:
    drake_factors = json.load(f)

with open(os.path.join(os.path.dirname(__file__), 'formationChannels.json')) as f:
    formation_channels = json.load(f)