import torch.nn as nn
import torch
import math
import numpy as np 

## Channel Attention (CA) Layer
class CALayer(nn.Module):
	def __init__(self, channel, reduction=16):
		super(CALayer, self).__init__()
		# global average pooling: feature --> point
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		# feature channel downscale and upscale --> channel weight
		self.conv_du = nn.Sequential(
				nn.Conv2d(channel, channel // reduction, 1 ,padding=0),
				nn.ReLU(inplace=True),
				nn.Conv2d(channel // reduction, channel, 1,padding=0),
				nn.Sigmoid()
		)

	def forward(self, x):
		y = self.avg_pool(x)
		y = self.conv_du(y)
		return x * y

		
class AttentionFusion(nn.Module):
	def __init__(self, channels, eps=1e-5):
		super(AttentionFusion, self).__init__()
		
		self.epsilon = eps
	
		self.alpha = nn.Parameter(torch.ones(1))
		self.gamma = nn.Parameter(torch.zeros(1))
		self.beta = nn.Parameter(torch.zeros(1))

	def forward(self,x):
		m_batchsize, N, C, height, width = x.size()
		x_reshape = x.view(m_batchsize, N, -1)
		M = C*height*width

		# compute covariance feature
		mean = torch.mean(x_reshape, dim=-1).unsqueeze(-1)
		x_reshape = x_reshape - mean
		cov = (1/(M-1) * x_reshape @ x_reshape.transpose(-1, -2))* self.alpha
		# print(cov)
		norm = cov/((cov.pow(2).mean((1,2),keepdim=True) + self.epsilon).pow(0.5))  # l-2 norm 

		attention = torch.tanh(self.gamma * norm + self.beta)
		x_reshape = x.view(m_batchsize, N, -1)

		out = torch.bmm(attention, x_reshape)
		out = out.view(m_batchsize, N, C, height, width)

		out += x
		out = out.view(m_batchsize, -1, height, width)
		return out, attention

		
## Residual Channel Attention Block (RCAB)
class ResidualBlock(nn.Module):
	def __init__(self,n_feat,kernel_size, stride, dilation, padding, bias=True):
		super(ResidualBlock, self).__init__()
		self.conv1 = nn.Conv2d(n_feat, n_feat, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding, bias=True)
		self.conv2 = nn.Conv2d(n_feat, n_feat, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding, bias=True)
		self.relu = nn.ReLU(inplace=True)
		# # initialization
		# initialize_weights([self.conv1, self.conv2], 0.1)
		self.CALayer = CALayer(n_feat, reduction=int(n_feat//4))
	def forward(self, x):
		out = self.relu(self.conv1(x))
		out = self.conv2(out)
		out = self.CALayer(out)
		return x + out

## Residual Group 
class ResidualGroup(nn.Module):
	def __init__(self ,n_blocks, n_feat, kernel_size, stride, dilation, padding, bias=True):
		super(ResidualGroup, self).__init__()

		self.fea_resblock = make_layer(ResidualBlock, n_feat, n_blocks,kernel_size, stride, dilation, padding)
		self.conv = nn.Conv2d(n_feat, n_feat,  kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding, bias=True)

	def forward(self, x):
		res = self.fea_resblock(x)
		res = self.conv(res)
		res += x
		return res

def default_conv(in_channels, out_channels, kernel_size, dilation, stride, padding, bias=True):
	return nn.Conv2d(
		in_channels, out_channels, kernel_size, stride=stride,
		 dilation =dilation, padding= padding, bias=bias)

class Upsampler(nn.Sequential):
	def __init__(self, scale, n_feat,kernel_size, stride, dilation, padding,  bn=False, act=False, bias=True):

		m = []
		if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
			for _ in range(int(math.log(scale, 2))):
				m.append(nn.Conv2d(n_feat, 4 * n_feat, kernel_size=kernel_size,stride=stride,dilation=dilation, padding=padding, bias=True))
				m.append(nn.PixelShuffle(2))
				if bn: m.append(nn.BatchNorm2d(n_feat))
				if act: m.append(act())
		elif scale == 3:
			m.append(nn.Conv2d(n_feat, 9 * n_feat, kernel_size=kernel_size,stride=stride,dilation=dilation, padding=padding, bias=True))
			m.append(nn.PixelShuffle(3))
			if bn: m.append(nn.BatchNorm2d(n_feat))
			if act: m.append(act())
		else:
			raise NotImplementedError

		super(Upsampler, self).__init__(*m)

def make_layer(block, nf, n_layers,kernel_size, stride, dilation, padding ):
	layers = []
	for _ in range(n_layers):
		layers.append(block(nf,kernel_size, stride, dilation, padding))
	return nn.Sequential(*layers)

def MacPI2SAI(x, angRes):
	out = []
	for i in range(angRes):
		out_h = []
		for j in range(angRes):
			out_h.append(x[:, :, i::angRes, j::angRes])
		out.append(torch.cat(out_h, 3))
	out = torch.cat(out, 2)
	return out

def MacPI2EPI(x, angRes):
	# N,C,W,H = x.size()

	train_data_0 = []
	train_data_90 = []
	train_data_45 = []
	train_data_135 = []

	index_center = int(angRes//2)
	for i in range(0,angRes,1):
		img_tmp = x[:,:,index_center::angRes,i::angRes]
		train_data_0.append(img_tmp)
	train_data_0 = torch.cat(train_data_0, 1)
	
	for i in range(0, angRes, 1):
		img_tmp = x[:,:,i::angRes, index_center::angRes]
		train_data_90.append(img_tmp)
	train_data_90 = torch.cat(train_data_90, 1)

	for i in range(0, angRes, 1):
		img_tmp = x[:,:,i::angRes, i::angRes]
		train_data_45.append(img_tmp)
	train_data_45 = torch.cat(train_data_45, 1)

	for i in range(0, angRes, 1):
		img_tmp = x[:,:,i::angRes, angRes-i-1::angRes]
		train_data_135.append(img_tmp)
	train_data_135 = torch.cat(train_data_135, 1)

	return train_data_0, train_data_90, train_data_45, train_data_135


def SAI24DLF(x, angRes):
	
	uh, vw = x.shape
	h0, w0 = int(uh // angRes), int(vw // angRes)

	LFout = torch.zeros(angRes, angRes, h0, w0)

	for u in range(angRes):
		start_u = u*h0
		end_u = (u+1)*h0
		for v in range(angRes):
			start_v = v*w0
			end_v = (v+1)*w0
			img_tmp = x[start_u:end_u,start_v:end_v]
			LFout[u, v, :, :] = img_tmp

	return LFout

def SAI2MacPI(x, angRes):
	b, c, hu, wv = x.shape
	h, w = hu // angRes, wv // angRes
	tempU = []
	for i in range(h):
		tempV = []
		for j in range(w):
			tempV.append(x[:, :, i::h, j::w])
		tempU.append(torch.cat(tempV, dim=3))
	out = torch.cat(tempU, dim=2)
	return out


