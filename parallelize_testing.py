from joblib import Parallel, delayed
def process(i):
    time.sleep(3)
    print(i)
    return i * i

# time the duration of the process

import time
print('Paralellized:')
start_time = time.time()
results = Parallel(n_jobs=4)(delayed(process)(i) for i in range(5))
print(results)
print("%s seconds" % (time.time() - start_time))

print('Non-paralellized:')
start_time = time.time()
for i in range(100000):
    process(i)

print("%s seconds" % (time.time() - start_time))