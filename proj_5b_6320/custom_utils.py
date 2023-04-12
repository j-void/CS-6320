import torch
import numpy as np
from numpy import asarray
from PIL import Image
from os import listdir, path
import matplotlib.pyplot as plt

def DataLoader(data_dir, verbose=False):
	l_images_dir = "image_2"
	r_images_dir = "image_3"
	gt_disp_dir = "disp_noc_0"
	Start_H = 150
	IM_H = 150 #1242
	IM_W = 1000 #375
	images_path = path.join(data_dir, gt_disp_dir)
	Image_Files = [f for f in listdir(images_path) if path.isfile(path.join(images_path,f))]
	x = torch.zeros([len(Image_Files), 1, 1, 2, 1, IM_H, IM_W])
	dispnoc = torch.zeros([len(Image_Files), IM_H, IM_W])
	l_images_path = path.join(data_dir, l_images_dir)
	r_images_path = path.join(data_dir, r_images_dir)
	gt_disp_path = path.join(data_dir, gt_disp_dir)
	for ctr, fn in enumerate(Image_Files):
		try:
			if fn.split(".")[0][-1] == '0':
				if verbose:
					print(f"Loading {fn}...")
				left = Image.open(path.join(l_images_path, fn)).convert('L')
				left = asarray(left)
				left = (left - np.mean(left))/np.std(left)
				left_t = torch.tensor(left)#.permute(2,0,1)
				#left_t = left_t - torch.mean(left_t)/torch.std(left_t)
				right = Image.open(path.join(r_images_path, fn)).convert('L')
				right = asarray(right)
				right = (right - np.mean(right))/np.std(right)
				right_t = torch.tensor(right)#.permute(2,0,1)
				#right_t = right_t - torch.mean(right_t)/torch.std(right_t)
				disp = Image.open(path.join(gt_disp_path, fn))
				disp_t = torch.tensor(asarray(disp))/256
				x[ctr][0][0][0][0] = left_t[Start_H:Start_H+IM_H, :IM_W]
				x[ctr][0][0][1][0] = right_t[Start_H:Start_H+IM_H, :IM_W]
				dispnoc[ctr] = disp_t[Start_H:Start_H+IM_H, :IM_W]
		except FileNotFoundError:
			print("All fields for record " + fn + " not found, skipping..." )
	print(f"Loaded {ctr+1} data points from {data_dir}")
	return x, dispnoc

def get_disparity(data, ind):
	x, disp = data
	left = x[ind][0][0][0][0]
	right = x[ind][0][0][1][0]
	gt = disp[ind]
	indices = torch.nonzero(gt, as_tuple=True)
	sel = np.random.randint(0, len(indices[0]))
	dim3 = indices[0][sel]
	dim4 = indices[1][sel]
	d = gt[dim3][dim4]
	return ind, dim3, dim4, d
	
def save_model(model, file_name):
	torch.save(model.state_dict(), file_name)

def load_model(model, file_name):
	model.load_state_dict(torch.load(file_name))
	return model

if __name__ == '__main__':
	X, disp = DataLoader("KITTI2015_Stereo")
	ind_img = 6
	#print(disp[6][17][668])
	print(get_disparity((X, disp), ind_img))
	print(get_disparity((X, disp), ind_img))
	print(get_disparity((X, disp), ind_img))
	print(get_disparity((X, disp), ind_img))
	print(get_disparity((X, disp), ind_img))
	print(get_disparity((X, disp), ind_img))
	#print(X[ind_img][0][0][0][0][::5][::5])
	#print('Left Image')
	#plt.imshow(X[ind_img][0][0][0][0].cpu().numpy(),cmap="gray")
	#plt.show()