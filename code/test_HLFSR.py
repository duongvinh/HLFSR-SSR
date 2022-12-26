import time
import argparse
import scipy.misc
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import os
from scipy import io

from common import *
from readH5 import *
from utils import *

from model_HLFSR import Net


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--device', type=str, default='cuda:0')
	parser.add_argument("--angRes", type=int, default=5, help="angular resolution")
	parser.add_argument("--upscale_factor", type=int, default=4, help="upscale factor")
	parser.add_argument("--channels", type=int, default=32, help="number of channels")
	parser.add_argument('--testset_dir', type=str, default='../x4/TestData_4xSR_5x5/')
	parser.add_argument("--crop_test_method",type=int, default=3, help="cropped test method( 1- whole image| 2- cropped mxn patches | 3- cropped 4 patches")
	parser.add_argument("--patchsize", type=int, default=64, help="LFs are cropped into patches to save GPU memory")
	parser.add_argument("--stride", type=int, default=32, help="The stride between two test patches is set to patchsize/2")
	parser.add_argument('--model_path', type=str, default='./log/HLFSR_4xSR_5x5_C32.pth.tar')
	parser.add_argument('--save_path', type=str, default='../Results/')

	return parser.parse_args()


def test(cfg, test_Names, test_loaders):

	net = Net(angRes=cfg.angRes, channels=cfg.channels, factor=cfg.upscale_factor) # HLFSR

	# net = Net(angRes=cfg.angRes, factor=cfg.upscale_factor) # DistgSSR 

	net.to(cfg.device)
	cudnn.benchmark = True

	# print(net)
	total_params = sum(p.numel() for p in net.parameters())
	print("Total Params: {:.2f}".format(total_params)) 

	if os.path.isfile(cfg.model_path):
		model = torch.load(cfg.model_path, map_location='cuda:0')
		net.load_state_dict(model['state_dict'])
	else:
		print("=> no model found at '{}'".format(cfg.load_model))

	with torch.no_grad():
		
		psnr_testset = []
		ssim_testset = []
		time_testset = []
		for index, test_name in enumerate(test_Names):
			test_loader = test_loaders[index]
			outLF, psnr_epoch_test, ssim_epoch_test, time_total = inference(test_loader, test_name, net,angRes=cfg.angRes)
			psnr_testset.append(psnr_epoch_test)
			ssim_testset.append(ssim_epoch_test)
			time_testset.append(time_total)
			print(time.ctime()[4:-5] + ' Valid----%15s, PSNR---%f, SSIM---%f, Time---%f ' % (test_name, psnr_epoch_test, ssim_epoch_test,time_total))
			pass
			
		psnr_mean_test = float(np.array(psnr_testset).mean())
		ssim_mean_test = float(np.array(ssim_testset).mean())
		time_testset = float(np.array(time_testset).mean())
		print('The mean psnr on testsets is %.5f, mean ssim is %.5f, Time---%f' % (psnr_mean_test, ssim_mean_test,time_testset))
		pass


def inference(test_loader, test_name, net,angRes=5,n_GPUs=1):
	psnr_iter_test = []
	ssim_iter_test = []
	avg_time_test = []

	for idx_iter, (data, label) in (enumerate(test_loader)):
		time_test = 0
		if cfg.crop_test_method == 1:
			data = data.to(cfg.device)  # numU, numV, h*angRes, w*angRes
			label = label.squeeze()
			with torch.no_grad():
				time_item_start = time.time()
				outLF = net(data)
				time_test += time.time() - time_item_start 
			outLF = outLF.squeeze()

		if cfg.crop_test_method == 2:
			data = MacPI2SAI(data,cfg.angRes)
			data = data.squeeze().to(cfg.device)  # numU, numV, h*angRes, w*angRes
			label = label.squeeze()

			uh, vw = data.shape
			h0, w0 = uh // cfg.angRes, vw // cfg.angRes
			subLFin = LFdivide(data, cfg.angRes, cfg.patchsize, cfg.stride)  # numU, numV, h*angRes, w*angRes
			numU, numV, H, W = subLFin.shape
			subLFout = torch.zeros(numU, numV, cfg.angRes * cfg.patchsize * cfg.upscale_factor, cfg.angRes * cfg.patchsize * cfg.upscale_factor)

			for u in range(numU):
				for v in range(numV):
					tmp = subLFin[u, v, :, :].unsqueeze(0).unsqueeze(0)
					tmp = SAI2MacPI(tmp,cfg.angRes)
					with torch.no_grad():
						time_item_start = time.time()
						torch.cuda.empty_cache()
						out = net(tmp.to(cfg.device))
						time_test += time.time() - time_item_start 
						subLFout[u, v, :, :] = out.squeeze()

			outLF = LFintegrate(subLFout, cfg.angRes, cfg.patchsize * cfg.upscale_factor, cfg.stride * cfg.upscale_factor, h0 * cfg.upscale_factor, w0 * cfg.upscale_factor)

		if cfg.crop_test_method == 3:
			data = data.to(cfg.device)  # numU, numV, h*angRes, w*angRes
			label = label.squeeze()

			b, c, h, w = data.size()
			scale  = cfg.upscale_factor
			h_half, w_half = h // 2, w // 2
			h_size, w_size = h_half , w_half 

			lr_list = [
				data[:, :, 0:h_size, 0:w_size],
				data[:, :, 0:h_size, (w - w_size):w],
				data[:, :, (h - h_size):h, 0:w_size],
				data[:, :, (h - h_size):h, (w - w_size):w]]


			sr_list = []
			for i in range(0, 4, n_GPUs):
				lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
				with torch.no_grad():
					time_item_start = time.time()
					sr_batch = net(lr_batch)
					time_test += time.time() - time_item_start 
				sr_batch = SAI2MacPI(sr_batch,angRes)
				sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))


			h, w = scale * h, scale * w
			h_half, w_half = scale * h_half, scale * w_half
			h_size, w_size = scale * h_size, scale * w_size
		
			outLF = data.new(b, c, h, w)
			outLF[:, :, 0:h_half, 0:w_half] \
				= sr_list[0][:, :, 0:h_half, 0:w_half]
			outLF[:, :, 0:h_half, w_half:w] \
				= sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
			outLF[:, :, h_half:h, 0:w_half] \
				= sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
			outLF[:, :, h_half:h, w_half:w] \
				= sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

			outLF = MacPI2SAI(outLF,angRes)
			outLF = outLF.squeeze()

		psnr, ssim = cal_metrics(label, outLF, cfg.angRes)
		psnr_iter_test.append(psnr)
		ssim_iter_test.append(ssim)
		avg_time_test.append(time_test)

		isExists = os.path.exists(cfg.save_path + test_name)
		if not (isExists ):
			os.makedirs(cfg.save_path + test_name)

		io.savemat(cfg.save_path + test_name + '/' + test_loader.dataset.file_list[idx_iter][0:-3] + '.mat',
						{'LF': outLF.cpu().numpy()})
		pass


	psnr_epoch_test = float(np.array(psnr_iter_test).mean())
	ssim_epoch_test = float(np.array(ssim_iter_test).mean())
	avg_time_test = float(np.array(avg_time_test).mean())

	return outLF, psnr_epoch_test, ssim_epoch_test,avg_time_test


def main(cfg):
	test_Names, test_Loaders, length_of_tests = MultiTestSetDataLoader(cfg)
	test(cfg, test_Names, test_Loaders)


if __name__ == '__main__':
	cfg = parse_args()
	main(cfg)
