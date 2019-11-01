"""
Author : Dharma Teja
"""

import torch
from torch.autograd import Function

enable_cuda = True


class ExpandBlockMask(Function):
	@staticmethod
	def forward(ctx, block_mask, bh, bw, kh=1, kw=1, is_fc=False):
		ctx.bh = bh
		ctx.bw = bw
		ctx.is_fc = is_fc

		nrb = block_mask.shape[0]
		ncb = block_mask.shape[1]

		mask = torch.ones(nrb*ncb, bh*bw*kh*kw, dtype=torch.float)

		if enable_cuda:
			device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
			mask = mask.to(device)
		
		if is_fc:
			trunc_block_mask = block_mask.clamp(0,1)
			f_trunc_block_mask = trunc_block_mask.view(nrb*ncb, -1)
			mask = mask * f_trunc_block_mask
			
			mask = mask.reshape(nrb, ncb, bh, bw)
			mask = mask.permute(0, 1, 3, 2)
			mask = mask.reshape(nrb, ncb*bw, bh)
			mask = mask.permute(0, 2, 1)
			mask = mask.reshape(nrb*bh, ncb*bw)
		else:
			trunc_block_mask = block_mask.clamp(0,1)
			f_trunc_block_mask = trunc_block_mask.view(nrb*ncb, -1)
			mask = mask * f_trunc_block_mask
			
			mask = mask.reshape(nrb, ncb, bh, bw, kh*kw)
			mask = mask.permute(0, 1, 3, 2, 4)
			mask = mask.reshape(nrb, ncb*bw, bh, kh*kw)
			mask = mask.permute(0, 2, 1, 3)
			mask = mask.reshape(nrb*bh, ncb*bw, kh, kw)

		return mask


	@staticmethod
	def backward(ctx, grad_output):
		bh = ctx.bh
		bw = ctx.bw

		rows = grad_output.shape[0]
		cols = grad_output.shape[1]

		nrb = rows // bh
		ncb = cols // bw
		

		grad_input = torch.zeros(nrb, ncb, dtype=torch.float)

		if enable_cuda:
			device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
			grad_input = grad_input.to(device)

		if ctx.is_fc:
			grad_output = grad_output.reshape(nrb, bh, cols)
			grad_output = grad_output.permute(0, 2, 1)
			grad_output = grad_output.reshape(nrb, ncb, bw, bh)
			grad_input = torch.sum(grad_output, dim=(2,3))

			return grad_input,None,None,None,None,None
		else:
			kh = grad_output.shape[-2]
			kw = grad_output.shape[-1]
			grad_output = grad_output.reshape(nrb, bh, cols, kh, kw)
			grad_output = grad_output.permute(0, 2, 1, 3, 4)
			grad_output = grad_output.reshape(nrb, ncb, bw, bh, kh, kw)
			grad_input = torch.sum(grad_output, dim=(2,3,4,5))

			return grad_input,None,None,None,None



from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _single, _pair, _triple
from torch.nn.parameter import Parameter
from torch.nn import init

import torch.nn.functional as F

from torch._jit_internal import weak_module, weak_script_method


@weak_module
class MaskedConv2d(_ConvNd):
	def __init__(self, in_channels, out_channels, kernel_size, stride=1,
		 padding=0, dilation=1, groups=1, bias=True, ogs=1, igs=1):

		kernel_size = _pair(kernel_size)
		stride = _pair(stride)
		padding = _pair(padding)
		dilation = _pair(dilation)
		super(MaskedConv2d, self).__init__(
			in_channels, out_channels, kernel_size, stride, padding, dilation,
			False, _pair(0), groups, bias)

		# New data
		self.ogs = ogs
		self.igs = igs
		num_ofm_groups = out_channels // ogs
		num_ifm_groups = in_channels  // igs

		self.blocked_mask = Parameter(torch.Tensor(num_ofm_groups, num_ifm_groups))
		self.reset_mask()

	def reset_mask(self):
		init.constant_(self.blocked_mask, 1.0)


	@weak_script_method
	def forward(self, input):
		expand_block_mask = ExpandBlockMask.apply
		mask = expand_block_mask(self.blocked_mask, 
					self.ogs, self.igs, 
					self.kernel_size[0], self.kernel_size[1])
		masked_weight = mask * self.weight
		return F.conv2d(input, masked_weight, self.bias, self.stride,
			self.padding, self.dilation, self.groups) 


import torch.nn as nn
import math

@weak_module
class MaskedLinear(nn.Module):
	__constants__ = ['bias']

	def __init__(self, in_features, out_features, bias=True, ogs=1, igs=1):
		super(MaskedLinear, self).__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.weight = Parameter(torch.Tensor(out_features, in_features))
		if bias:
			self.bias = Parameter(torch.Tensor(out_features))
		else:
			self.register_parameter('bias', None)

		# New data
		self.ogs = ogs
		self.igs = igs
		num_ofm_groups = out_features // ogs
		num_ifm_groups = in_features // igs

		self.blocked_mask = Parameter(torch.Tensor(num_ofm_groups, num_ifm_groups))
		self.reset_parameters()

	def reset_parameters(self):
		init.constant_(self.blocked_mask, 1.0) 

	@weak_script_method
	def forward(self, input):
		expand_block_mask = ExpandBlockMask.apply
		mask = expand_block_mask(self.blocked_mask, 
					self.ogs, self.igs, 
					1, 1, True)
		masked_weight = mask * self.weight
		return F.linear(input, masked_weight, self.bias)

	def extra_repr(self):
		return 'in_features={}, out_features={}, bias={}'.format(
			self.in_features, self.out_features, self.bias is not None
		)


if __name__ == "__main__":
	"""
	ofm = 10
	ifm = 12
	ksize = 1

	imsize = ksize
	ogs = 2
	igs = 3

	conv1 = MaskedConv2d(ifm, ofm, ksize, bias=False, 
				igs=igs, ogs=ogs)
	#input = torch.randint(1,5, shape=(ifm, imsize, imsize), dtype=torch.float)
	input = torch.ones(1, ifm, imsize, imsize)

	print(input)
	print(conv1.weight)
	print(conv1.blocked_mask)
	output = conv1(input)

	print(output)

	import sys
	sys.exit(-1)
	"""

	bh = 2
	bw = 3
	rows = 10
	cols = 12

	nrb = rows//bh
	ncb = cols//bw 


	expand_block_mask = ExpandBlockMask.apply

	b_mask = torch.zeros(nrb, ncb, requires_grad=True)
	init.uniform_(b_mask, 0, 1)
	w = torch.randint(1,3, size=(rows,cols), requires_grad=True, dtype=torch.float)
	x = torch.randint(1,3, size=(cols,1), requires_grad=True, dtype=torch.float)

	# Computation
	mask = expand_block_mask(b_mask, bh, bw , 1, 1, True)
	wm = mask * w
	y = torch.matmul(wm, x)

	print(b_mask)
	print(mask)
	print(w)
	print(x)
	print(y)

	dy = torch.randint(1,3, size=(rows,1), dtype=torch.float)
	y.backward(dy)

	print(dy)
	wm_grad = torch.matmul(dy, x.transpose(1,0))
	w_grad  = mask * wm_grad
	mask_grad  = w * wm_grad

	b_mask_grad = torch.zeros(nrb, ncb)
	for rb_id in range(nrb):
			for cb_id in range(ncb):
				b_mask_grad[rb_id, cb_id] = torch.sum(mask_grad[(rb_id*bh):(rb_id+1)*bh, cb_id*bw:(cb_id+1)*bw])

	print(w_grad)
	print(w.grad)
	print(mask_grad)
	print(b_mask.grad)

	assert(w_grad == w.grad).all()
	assert(b_mask_grad == b_mask.grad).all()
			
