import torch
import torchvision
from PIL import Image
from 一眼丁真.code.data import word_list1, word_list2
from 一眼丁真.code.show_result import show

# 加载网络模型
model1 = torch.load('model/network_1.pth')
model2 = torch.load('model/network_2.pth')

# 输入图片
try:
    i = input('请选择需要鉴定的图片名称：')
    img_path = 'img/{}.jpg'.format(i)
    img = Image.open(img_path)
except:
    print("未找到指定图片，请确认图片名称！已使用样例图片进行鉴定！")
    img = Image.open('img/sample.jpg')

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])
image = transform(img)
image = torch.reshape(image, (1, 3, 32, 32))

# 使用卷积网络测试
model1.eval()
model2.eval()
with torch.no_grad():
    output1 = model1(image)
    output2 = model2(image)

# 输出测试结果
word_list1[0] = ''
out = '一眼丁真，鉴定为：{}{}'.format(word_list1[output1.argmax(1).item()], word_list2[output2.argmax(1).item()])
print(out)
show(img, out)
