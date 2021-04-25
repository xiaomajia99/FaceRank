import os
import copy
import time
import numpy as np
import logging

logging.basicConfig(filename='logger.log', level=logging.INFO)

logging.info("123424")


test = {"123": 0.32, "345": 0.54}

if os.path.exists('./xx.csv'):
    pass
else:
    pass


with open("./xx.csv", 'w') as hh:
    for k,v in test.items():
        hh.write(k + "," + str(v) + "\n")

xx = list(test.keys())
yy = copy.deepcopy(xx)

print(id(xx), id(yy))


xx = range(1,10000000)
yy = range(10000,20000)

print(np.array(yy))
start_time = time.time()
for y in yy:
    if y in xx:
        pass
    else:
        pass
end_time = time.time()
print(end_time - start_time)
    