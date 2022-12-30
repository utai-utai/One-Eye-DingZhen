import matplotlib
import matplotlib.pyplot as plt


# 使用plt输出图片和文字
def show(image, word):
    plt.imshow(image)
    matplotlib.rcParams['font.sans-serif'] = ['KaiTi']
    plt.title(word, fontsize=20)
    plt.xticks([])
    plt.yticks([])
    plt.show()
