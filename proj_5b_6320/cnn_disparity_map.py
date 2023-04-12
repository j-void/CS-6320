"""
This file holds the main code for disparity map calculations
"""
import torch
import numpy as np

from typing import Callable, Tuple


def cnn_calculate_disparity_map(left_img: torch.Tensor,
                            right_img: torch.Tensor,
                            block_size: int,
                            cost_volume: torch.Tensor,
                            max_search_bound: int = 50) -> torch.Tensor:
	assert left_img.shape == right_img.shape
	disparity_map = torch.zeros(1) #placeholder, this is not the actual size
	H = left_img.shape[0]
	W = left_img.shape[1]
	block_r = block_size//2  # Block (Window) radius
	H_d = H-2*(block_r)
	W_d = W-2*(block_r)

	disparity_map = torch.zeros(H_d, W_d)
	for y in range(H_d):
		for x in range(W_d):
			min_err = 500
			min_d = torch.argmin(cost_volume[y,x])
			disparity_map[y,x] = min_d
	return disparity_map

def cnn_calculate_cost_volume(left_img: torch.Tensor,
                          right_img: torch.Tensor,
                          max_disparity: int,
                          cnn_batch_sim_function: callable,
                          block_size: int = 9):
	#placeholder
	H = left_img.shape[0]
	W = right_img.shape[1]
	block_r = block_size//2
	cost_volume = torch.zeros(H, W, max_disparity) + 255

	for y in range(H):
		print("Done row: ",y)
		for x in range(W):

			y_l = y-block_r
			x_l = x-block_r

			# Ensure we can extract a full patch
			if (x_l < 0 or y_l < 0 or x_l > W-block_size or y_l > H-block_size):
				cost_volume[y,x,:] = 255
				continue
      
			# Extract patch starting at point (y,x) in left image
			left_patch = left_img[y_l:y_l+block_size, x_l:x_l+block_size,0]
			right_strip = right_img[y_l:y_l+block_size, 0:x_l+block_size,0]
			max_range = min(max_disparity,x_l)
			res = torch.tensor(cnn_batch_sim_function(left_patch, right_strip)[:])
			cost_volume[y,x,:res.shape[0]] = res.T
	return cost_volume
