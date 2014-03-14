import json
import numpy
import matplotlib.pyplot as plt

a = []
with open('/tmp/a') as ins:
    for line in ins:
        a.append(json.loads(line))
b = []
with open('/tmp/b') as ins:
    for line in ins:
        b.append(json.loads(line))
inputs = []
with open('/tmp/input') as ins:
    for line in ins:
        inputs.append(json.loads(line))

plt.ioff(); plt.clf(); plt.hold(1);

inputhl, = plt.plot([], [])
ahl, = plt.plot([], [])
b0hl, = plt.plot([], [])
b1hl, = plt.plot([], [])
b2hl, = plt.plot([], [])
plt.legend(['Input','A','B0','B1','B2'])
plt.show(block=False)
plt.tight_layout()

plt.xlim((0, 500))
plt.ylim((-1.5, 1.5))


def update_line(hl, x, new_data):
    hl.set_xdata(numpy.append(hl.get_xdata(), x))
    hl.set_ydata(numpy.append(hl.get_ydata(), new_data))

import time
draw = 0
for x in range(len(a)):
    update_line(ahl, x, a[x][0])
    update_line(b0hl, x, b[x][0])
    update_line(b1hl, x, b[x][1])
    update_line(b2hl, x, b[x][2])
    update_line(inputhl, x, inputs[x][0])
    draw += 1
    if draw is 4:
        plt.draw()
        draw = 0
    time.sleep(0.001)

time.sleep(10)

