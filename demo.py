import json
import matplotlib.pyplot as plt

array = []
with open('/tmp/a') as ins:
    for line in ins:
        array.append(json.loads(line))

plt.ioff(); plt.clf(); plt.hold(1);
plt.legend(['A'])
plt.show(block=False)
plt.tight_layout()
plt.xlim((0, 500))
plt.ylim((-1.5, 1.5))

def update_plot(arr):
    plt.plot(arr)

import time
plot_data = []
draw = 0
for x in array:
    plot_data.append(x)
    update_plot(plot_data)
    draw += 1
    if draw is 5:
        plt.draw()
        draw = 0
    time.sleep(0.005)

