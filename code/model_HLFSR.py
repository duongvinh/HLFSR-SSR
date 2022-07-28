import torch
import torch.nn as nn
from common import * 
import torch.nn.functional as F

class HLFSR(nn.Module):
	def __init__(self, angRes, n_blocks, channels, upscale_factor):
		super(HLFSR, self).__init__()
		
		self.angRes = angRes
		self.n_blocks = n_blocks
		self.upscale_factor = upscale_factor
	   
		
		self.HFEM_1 = HFEM(angRes, n_blocks, channels,first=True)
		self.HFEM_2 = HFEM(angRes, n_blocks, channels,first=False)
		self.HFEM_3 = HFEM(angRes, n_blocks, channels,first=False)
		self.HFEM_4 = HFEM(angRes, n_blocks, channels,first=False)
		self.HFEM_5 = HFEM(angRes, n_blocks, channels,first=False)

		# define tail module for upsamling
		UpSample = [
			Upsampler(self.upscale_factor, channels,kernel_size=3, stride =1,dilation=1, padding=1, act=False),
			nn.Conv2d(channels, 1, kernel_size=1,stride =1,dilation=1,padding=0,bias=False)]
	
		self.UpSample = nn.Sequential(*UpSample)
		

	def forward(self, x):
		
		#Reshaping and Upscaling
		x_reshaping = MacPI2SAI(x,self.angRes)
		x_upscale = F.interpolate(x_reshaping, scale_factor=self.upscale_factor, mode='bicubic', align_corners=False)

		HFEM_1 = self.HFEM_1(x)
		HFEM_2 = self.HFEM_2(HFEM_1)
		HFEM_3 = self.HFEM_3(HFEM_2)
		HFEM_4 = self.HFEM_4(HFEM_3)
		HFEM_5 = self.HFEM_5(HFEM_4)


		#Reshaping
		x_out = MacPI2SAI(HFEM_5,self.angRes)
		x_out= self.UpSample(x_out)

		x_out += x_upscale

		return x_out


class HFEM(nn.Module):
	def __init__(self, angRes, n_blocks, channels,first=False):
		super(HFEM, self).__init__()
		
		self.first = first 
		self.n_blocks = n_blocks
		self.angRes = angRes

		# define head module epi feature
		head_epi = []
		if first:  
			head_epi.append(nn.Conv2d(angRes, channels, kernel_size=3, stride=1, padding=1, bias=False))
		else:
			head_epi.append(nn.Conv2d(angRes*channels, channels, kernel_size=3, stride=1, padding=1, bias=False))

		self.head_epi = nn.Sequential(*head_epi)

		self.epi2spa = nn.Sequential(
			nn.Conv2d(4*channels, int(angRes * angRes * channels), kernel_size=1, stride=1, padding=0, bias=False),
			nn.PixelShuffle(angRes),
		)


		# define head module intra spatial feature
		head_spa_intra = []
		if first:  
			head_spa_intra.append(nn.Conv2d(1 ,channels, kernel_size=3, stride=1,dilation=int(angRes), padding=int(angRes), bias=False))
			
		else:
			head_spa_intra.append(nn.Conv2d(channels ,channels, kernel_size=3, stride=1,dilation=int(angRes), padding=int(angRes), bias=False))
			

		self.head_spa_intra = nn.Sequential(*head_spa_intra)


		# define head module inter spatial feature
		head_spa_inter = []
		if first:  
			head_spa_inter.append(nn.Conv2d(1 ,channels, kernel_size=3, stride=1,dilation=1, padding=1, bias=False))
		else:
			head_spa_inter.append(nn.Conv2d(channels ,channels, kernel_size=3, stride=1,dilation=1, padding=1, bias=False))
			

		self.head_spa_inter = nn.Sequential(*head_spa_inter)

		

		# define head module intra angular feature
		head_ang_intra = []
		if first: 
			head_ang_intra.append(nn.Conv2d(1 ,channels, kernel_size=int(angRes), stride = int(angRes), dilation=1, padding=0, bias=False))

		else:
			head_ang_intra.append(nn.Conv2d(channels ,channels, kernel_size=int(angRes), stride = int(angRes), dilation=1, padding=0, bias=False))
			

		self.head_ang_intra = nn.Sequential(*head_ang_intra)

		self.ang2spa_intra = nn.Sequential(
			nn.Conv2d(channels, int(angRes * angRes * channels), kernel_size=1, stride=1, padding=0, bias=False),
			nn.PixelShuffle(angRes), 
		)


		# define head module inter angular feature
		head_ang_inter = []
		if first:  
			head_ang_inter.append(nn.Conv2d(1 ,channels, kernel_size=int(angRes*2), stride = int(angRes*2), dilation=1, padding=0, bias=False))

		else:
			head_ang_inter.append(nn.Conv2d(channels ,channels, kernel_size=int(angRes*2), stride = int(angRes*2), dilation=1, padding=0, bias=False))
			

		self.head_ang_inter = nn.Sequential(*head_ang_inter)

			
		self.ang2spa_inter = nn.Sequential(
			nn.Conv2d(channels, int(4*angRes * angRes * channels), kernel_size=1, stride=1, padding=0, bias=False),
			nn.PixelShuffle(2*angRes),
		)

		# define  module attention fusion feature
		self.attention_fusion =  AttentionFusion(channels)
											
		# define  module spatial residual group
		self.SRG = nn.Sequential( nn.Conv2d(5*channels, channels, kernel_size=1,stride =1,dilation=1,padding=0, bias=False),
		                          ResidualGroup(self.n_blocks, channels,kernel_size=3,stride =1,dilation=int(angRes), padding=int(angRes), bias=False))


	def forward(self, x):

		# MO-EPI feature extractor
		data_0, data_90, data_45, data_135 = MacPI2EPI(x,self.angRes)

		data_0 = self.head_epi(data_0)
		data_90 = self.head_epi(data_90)
		data_45 = self.head_epi(data_45)
		data_135 = self.head_epi(data_135)
	
		mid_merged = torch.cat((data_0, data_90, data_45, data_135), 1)
		x_epi = self.epi2spa(mid_merged)


		# intra/inter spatial feature extractor
		x_s_intra = self.head_spa_intra(x)
		x_s_inter = self.head_spa_inter(x)
	
		# intra/inter angular feature extractor
		x_a_intra = self.head_ang_intra(x)
		x_a_intra = self.ang2spa_intra(x_a_intra)

		x_a_inter = self.head_ang_inter(x)
		x_a_inter = self.ang2spa_inter(x_a_inter)

		# fusion feature and refinement
		out = x_s_intra.unsqueeze(1)
		out = torch.cat([x_s_inter.unsqueeze(1),out],1)
		out = torch.cat([x_a_intra.unsqueeze(1),out],1)
		out = torch.cat([x_a_inter.unsqueeze(1),out],1)
		out = torch.cat([x_epi.unsqueeze(1),out],1)

		out,att_weight = self.attention_fusion(out)

		out = self.SRG(out)

		return out






if __name__ == "__main__":
    net = HLFSR(5,5,64, 2).cuda()
    from thop import profile
    input = torch.randn(1, 1, 160, 160).cuda()
    total = sum([param.nelement() for param in net.parameters()])
    flops, params = profile(net, inputs=(input,))
    print('   Number of parameters: %.2fM' % (total / 1024**2))
    print('   Number of FLOPs: %.2fG' % (flops / 1024**3))
