''' Image Source Interface Class'''

import numpy as np


class Source:
    def getNextFrame(self) -> np.array:
        raise NotImplementedError()
