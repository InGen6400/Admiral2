import queue
from collections import deque

a = deque(maxlen=5)
a.append(1)
a.append(2)
a.append(3)
a.append(4)
a.append(5)
a.append(6)
a.append(7)

print(a)
print(sum(a))
print(len(a))
