#coding:utf-8
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.utils import save_image

import numpy as np
import os
#import paras
import pandas as pd
from PIL import Image
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

# 超参数
gpu_id = None
if gpu_id is not None:
	os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
	device = torch.device('cuda')
else:
	device = torch.device('cpu')
if os.path.exists('cgan_images') is False:
	os.makedirs('cgan_images')

z_dim = 100 #paras.z_dim
batch_size = 100 #paras.batch_size
learning_rate = 0.0003 #paras.learning_rate
total_epochs = 10 #paras.total_epochs

class Discriminator(nn.Module):
	def __init__(self):
		super().__init__()
		self.model = nn.Sequential(
			nn.Linear(794, 1024),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Dropout(0.3),
			nn.Linear(1024, 512),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Dropout(0.3),
			nn.Linear(512, 256),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Dropout(0.3),
			nn.Linear(256, 1),
			nn.Sigmoid()
		)

	def forward(self, x, c):
		x = x.view(x.size(0), 784)
		x = torch.cat([x, c], 1)
		out = self.model(x)
		return out.squeeze()


class Generator(nn.Module):
	def __init__(self, z_dim):
		super().__init__()
		self.model = nn.Sequential(
			nn.Linear(10 + z_dim, 256),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(256, 512),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(512, 1024),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(1024, 784),
			nn.Tanh()
		)

	def forward(self, z, c):
		z = z.view(z.size(0), z_dim)
		x = torch.cat([z, c], 1)
		out = self.model(x)
		return out.view(x.size(0), 28, 28)


def one_hot(labels, class_num):
	batch_size = labels.size(0)
	one_hot_label = torch.zeros(batch_size, class_num).scatter_(1, labels.reshape(-1, 1), 1)
	return one_hot_label


# 初始化构建判别器和生成器
discriminator = Discriminator().to(device)
generator = Generator(z_dim=z_dim).to(device)

# 初始化二值交叉熵损失
bce = torch.nn.BCELoss().to(device)
ones = torch.ones(batch_size).to(device)
zeros = torch.zeros(batch_size).to(device)

# 初始化优化器，使用Adam优化器
g_optimizer = optim.Adam(generator.parameters(), lr=learning_rate, betas=[0.5, 0.999])
d_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=[0.5, 0.999])

# 加载fashion数据集

#############################################################
class FashionMNIST(Dataset):
	def __init__(self, transform=None):
		self.transform = transform
		fashion_df = pd.read_csv('fashionmnist/fashion-mnist_train.csv')
		self.labels = fashion_df.label.values
		self.images = fashion_df.iloc[:, 1:].values.astype('uint8').reshape(-1, 28, 28)

	def __len__(self):
		return len(self.images)

	def __getitem__(self, idx):
		label = self.labels[idx]
		img = self.images[idx]
		img = Image.fromarray(self.images[idx])

		if self.transform:
			img = self.transform(img)

		return img[0].resize(1, 28, 28), label

transform = transforms.Compose([
        transforms.ToTensor(),
		transforms.Lambda(lambda x: x.repeat(3,1,1)),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
dataset = FashionMNIST(transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

def show_sample(idx):
	sample = np.array(dataset[idx][0])
	sample = torch.from_numpy(sample)
	grid = make_grid(sample.unsqueeze(1).data, nrow=1, normalize=True).permute(1, 2, 0).numpy()
	plt.imshow(grid)
	plt.show()

import random
for i in range(10):
	a = random.randint(0, 10000)
	show_sample(a)
#############################################################

#用于生成效果图
# 生成100个one_hot向量，每类10个
fixed_c = torch.FloatTensor(100, 10).zero_()
fixed_c = fixed_c.scatter_(dim=1, index=torch.LongTensor(np.array(np.arange(0, 10).tolist()*10).reshape([100, 1])), value=1)
fixed_c = fixed_c.to(device)
# 生成100个随机噪声向量
fixed_z = torch.randn([100, z_dim]).to(device)

# 开始训练，一共训练total_epochs
for epoch in range(total_epochs):

	# 在训练阶段，把生成器设置为训练模式；对应于后面的，在测试阶段，把生成器设置为测试模式
	generator = generator.train()

	# 训练一个epoch
	for i, data in enumerate(dataloader):

		# 加载真实数据
		###############################
		img, label = data
		###############################

		# 把对应的标签转化成 one-hot 类型
		################################
		label = one_hot(label, 10)
		################################

		# 生成数据
		# 用正态分布中采样batch_size个随机噪声
		z = torch.randn([batch_size, z_dim]).to(device)
		# 生成 batch_size 个 ont-hot 标签
		c = torch.FloatTensor(batch_size, 10).zero_()
		c = c.scatter_(dim=1, index=torch.LongTensor(np.array(np.arange(0, 10).tolist() * 10).reshape([batch_size, 1])), value=1)
		c = c.to(device)
		# 生成数据

		#your code
		fake_img = generator(z, c)

		# 计算判别器损失，并优化判别器

		#######################################
		real_prob = discriminator(img, label)
		real_loss = bce(real_prob, ones)

		fake_prob = discriminator(fake_img, c)
		fake_loss = bce(fake_prob, zeros)

		d_optimizer.zero_grad()
		d_loss = real_loss + fake_loss
		d_loss.backward()
		d_optimizer.step()
		#######################################

		# 计算生成器损失，并优化生成器

		#######################################
		#your code
		g_optimizer.zero_grad()
		z = torch.randn(batch_size, z_dim)
		c = torch.LongTensor(np.random.randint(0, 10, batch_size))
		c = one_hot(c, 10)
		fake_img = generator(z, c)
		prob = discriminator(fake_img, c)
		g_loss = bce(prob, torch.ones(batch_size))
		g_loss.backward()
		g_optimizer.step()

		#######################################

		# 输出损失 参考下方 print
		print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, total_epochs, i, len(dataloader), d_loss.item(), g_loss.item()))

	# 把生成器设置为测试模型，生成效果图并保存
	generator = generator.eval()
	fixed_fake_images = generator(fixed_z, fixed_c)
	grid = make_grid(fixed_fake_images.unsqueeze(1).data, nrow=10, normalize=True).permute(1, 2, 0).numpy()
	plt.imshow(grid)
	plt.show()

	#save_image(fixed_fake_images, 'cgan_images/{}.png'.format(epoch), nrow=10, normalize=True)

