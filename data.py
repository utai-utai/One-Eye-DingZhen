import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

word_list1 = ['？', '春春的', '纯纯的', '滴能', '司马', '青年', '砸钟', '大']
word_list2 = ['厨神', '傻卵', '滴嫩儿', '桂梧', '白齿', '纸张', '栽种', '杂种', '人妖', '飞物', '傻逼',
              '脑瘫', '贱货', '函件', '消愁', '铸币']


class MyData(Dataset):
    def __init__(self, path, transform):
        self.path = path
        self.transform = transform
        self.images = os.listdir(self.path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_index = self.images[index]
        img_path = os.path.join(self.path, img_index)
        img = Image.open(img_path).convert('RGB')
        # 获得第一个标签
        label_1 = img_path.split('\\')[-1].split('.')[1]
        result_1 = 0
        for i in range(len(word_list1)):
            if word_list1[i] == label_1:
                result_1 = i
        # 获得第二个标签
        label_2 = img_path.split('\\')[-1].split('.')[2]
        result_2 = 0
        for i in range(len(word_list2)):
            if word_list2[i] == label_2:
                result_2 = i

        if self.transform is not None:
            img = self.transform(img)

        return img, result_1, result_2


def dataset():
    batch_size = 1024
    transform = transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor(),
                                    transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])])
    train_dataset = MyData(path='../data', transform=transform)
    print('训练数据集大小为{}'.format(len(train_dataset)))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_dataloader
