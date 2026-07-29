from dev.helpers import *

import matplotlib.pyplot as plt
from pathlib import Path
path = Path(__file__).parent.parent.parent / "academic-mplstyle/nice.mplstyle"
plt.style.use(path)
