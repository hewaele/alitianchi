import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# plt.ion()
fig, ax = plt.subplots()

x = np.arange(0, 2*np.pi, 0.01)
line, = ax.plot(x, np.sin(x))

#定义init_func,给定初始信息
def init():  # only required for blitting to give a clean slate.
    line.set_ydata([np.nan] * len(x))
    return line,

#定义func,用来反复调用的函数
def animate(i):
    line.set_ydata(np.sin(x + i / 100))  # 跟随自变量的增加更新y值
    return line,


ani = animation.FuncAnimation(
    fig, animate, init_func=init, interval=2, blit=True, save_count=50, repeat=False)
plt.show()

#
print('输出测试啊')

def drawanimationhist():
    fig, ax = plt.subplots()
    BINS = np.linspace(-5, 5, 100)
    data = np.random.randn(1000)
    n, _ = np.histogram(data, BINS)
    _, _, bar_container = ax.hist(data, BINS, lw=2,
                                  ec="b", fc="pink")
    def update(bar_container):
        def animate(frame_number):
            data = np.random.randn(1000)
            n, _ = np.histogram(data, BINS)
            for count, rect in zip(n, bar_container.patches):
                rect.set_height(count)
            return bar_container.patches
        return animate

    ax.set_ylim(top=55)

    ani = animation.FuncAnimation(fig, update(bar_container), 50,
                                  repeat=False, blit=True)
    plt.show()

drawanimationhist()
# time.sleep(60)