import numpy as np
from game import Connect4Game

board = np.array([[0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 42, 0, 0],
                  [0, 0, 0, 0, -1, 0, 0]])

batch_idx = 0
batch_size = 3
examples = [1,2,3,4,5,6,7,8,9,10,11,12]
print(examples[len(examples)-1])

while batch_idx < int(len(examples) / batch_size):
    # randomly batch train examples
    #print(list(range(batch_idx*batch_size,(batch_idx+1)*batch_size) ))
    sample_ids = np.random.randint(len(examples), size=batch_size)
    
    #boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
    batch_idx+=1