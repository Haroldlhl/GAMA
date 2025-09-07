import sys
import numpy as np


a = "{'train':[[1,1,5],[1,2,3],[2,1,4],[2,2,2]], 'test':[[1,2],[2,3],[3,1]]}"
data = eval(a)
history = np.array(data['train'])
tests = np.array(data['test'])
print(tests)


ans = []
all_mean = np.mean(history[:, 2])
# 如果test 在 history[:, :2] 中， 则输出history[idx, 2] 
# 如果test[0] 在 history[:, 0] 中， 则输出history[idxs, 2]的均值
# 否者输出全局均值


