import numpy as np
import pandas as pd
data1 = [1, 2, 3, 4, 5]
data2 = [[1, 3, 4], [2, 5, 6]]
array2 = np.array(data2)
array1 = np.array(data1)

pd_series = pd.Series(data1)
print(pd_series)
