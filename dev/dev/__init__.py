import os

import matplotlib.pyplot as plt

from dev.helpers import *
from .datapaths import DATAPATHS, SNAPSHOT_TFB

path = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "academic-mplstyle", "nice.mplstyle")
)
plt.style.use(path)
