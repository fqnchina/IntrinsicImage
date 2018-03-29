#include "common.h"

template<typename Dtype>
__global__ void computeWeights_impro(const int num_kernels, Dtype* edge,
		Dtype* weight, const int plane, const int height, const int width,
		const float sigma_k, const float div) {
	CUDA_KERNEL_LOOP(index, num_kernels)
	{
		weight = weight + index;
		Dtype grad = *(edge + index);
		*weight = exp(sigma_k * (1 + grad * div));
		// *weight = grad;
		*weight = *weight > 1 ? 1 : *weight;
		*weight = *weight < 0 ? 0 : *weight;
	}
}

template<typename Dtype>
__global__ void leftToRight_impro(const int num_kernels, Dtype* weight,
		Dtype* output, Dtype* inter, const int height, const int width) {
	CUDA_KERNEL_LOOP(index, num_kernels)
	{
		int h = index % height;
		int c = index / height;
		bool b2w = false;
		int count = 0;
		for (int w = 1; w < width; w++) {
			int out_idx = (c * height + h) * width + w;
			int wei_idx = h * width + w;
			*(inter + out_idx) = *(output + out_idx - 1) - *(output + out_idx);
//			if (*(weight + wei_idx) < 0.2)
//				b2w = true;
//			if (*(weight + wei_idx) > 0.8 && b2w) {
//				count++;
//				continue;
//			}
//			if (count == 3) {
//				count = 0;
//				b2w = false;
//			}
			*(output + out_idx) += *(weight + wei_idx) * (*(inter + out_idx));
		}
	}
}

template<typename Dtype>
__global__ void rightToLeft_impro(const int num_kernels, Dtype* weight,
		Dtype* output, Dtype* inter, const int height, const int width) {
	CUDA_KERNEL_LOOP(index, num_kernels)
	{
		int h = index % height;
		int c = index / height;
		for (int w = width - 2; w >= 0; w--) {
			int out_idx = (c * height + h) * width + w;
			int wei_idx = h * width + w;
			*(inter + out_idx) = *(output + out_idx + 1) - *(output + out_idx);
			*(output + out_idx) += *(weight + wei_idx + 1)
					* (*(inter + out_idx));
		}
	}
}

template<typename Dtype>
__global__ void topToBottom_impro(const int num_kernels, Dtype* weight,
		Dtype* output, Dtype* inter, const int height, const int width) {
	CUDA_KERNEL_LOOP(index, num_kernels)
	{
		int w = index % width;
		int c = index / width;
		for (int h = 1; h < height; h++) {
			int out_idx = (c * height + h) * width + w;
			int wei_idx = h * width + w;
			*(inter + out_idx) = *(output + out_idx - width)
					- *(output + out_idx);
			*(output + out_idx) += *(weight + wei_idx) * (*(inter + out_idx));
		}
	}
}

template<typename Dtype>
__global__ void bottomToTop_impro(const int num_kernels, Dtype* weight,
		Dtype* output, Dtype* inter, const int height, const int width) {
	CUDA_KERNEL_LOOP(index, num_kernels)
	{
		int w = index % width;
		int c = index / width;
		for (int h = height - 2; h >= 0; h--) {
			int out_idx = (c * height + h) * width + w;
			int wei_idx = h * width + w;
			*(inter + out_idx) = *(output + out_idx + width)
					- *(output + out_idx);
			*(output + out_idx) += *(weight + wei_idx + width)
					* (*(inter + out_idx));
		}
	}
}

template<typename Dtype>
void domainTransform_impro(cudaStream_t stream, Dtype* edge, Dtype* weight,
		Dtype* output, Dtype* inter, const int plane, const int height,
		const int width, const int num_iter, const float sigma_range,
		const int sigma_spatial) {

	for (int m = 0; m < num_iter; m++) {
		float sigma_k = sqrt(3 / (pow(4, num_iter) - 1)) * sigma_spatial
				* pow(2, num_iter - m - 1);
		sigma_k = -sqrt(2) / sigma_k;
		float div = sigma_spatial / sigma_range;
		Dtype* weight_m = weight + m * height * width;
		Dtype* inter_m = inter + m * height * width * plane * 4;

		int dimSize = 1024;
		int num_kernels = height * width;
		int grid = (num_kernels + dimSize - 1) / dimSize;
		computeWeights_impro<<<grid, dimSize, 0, stream>>>(num_kernels, edge,
				weight_m, plane, height, width, sigma_k, div);

		dimSize = 1024;
		num_kernels = height * plane;
		grid = (num_kernels + dimSize - 1) / dimSize;
		leftToRight_impro<<<grid, dimSize, 0, stream>>>(num_kernels, weight_m,
				output, inter_m, height, width);
		inter_m += height * width * plane;
		rightToLeft_impro<<<grid, dimSize, 0, stream>>>(num_kernels, weight_m,
				output, inter_m, height, width);

		dimSize = 1024;
		num_kernels = width * plane;
		grid = (num_kernels + dimSize - 1) / dimSize;
		inter_m += height * width * plane;
		topToBottom_impro<<<grid, dimSize, 0, stream>>>(num_kernels, weight_m,
				output, inter_m, height, width);
		inter_m += height * width * plane;
		bottomToTop_impro<<<grid, dimSize, 0, stream>>>(num_kernels, weight_m,
				output, inter_m, height, width);
	}
}

template<typename Dtype>
__global__ void bottomToTop_backward_impro(const int num_kernels,
		Dtype* gradData, Dtype* weight, Dtype* inter, Dtype* gradWeight,
		const int height, const int width) {
	CUDA_KERNEL_LOOP(index, num_kernels)
	{
		int w = index % width;
		int c = index / width;
		for (int h = 0; h < height - 1; h++) {
			int out_idx = (c * height + h) * width + w;
			int wei_idx = h * width + w;
			*(gradWeight + wei_idx + width) += *(gradData + out_idx)
					* (*(inter + out_idx));
			*(gradData + out_idx + width) += *(weight + wei_idx + width)
					* (*(gradData + out_idx));
			*(gradData + out_idx) *= 1 - *(weight + wei_idx + width);
		}
	}
}

template<typename Dtype>
__global__ void topToBottom_backward_impro(const int num_kernels,
		Dtype* gradData, Dtype* weight, Dtype* inter, Dtype* gradWeight,
		const int height, const int width) {
	CUDA_KERNEL_LOOP(index, num_kernels)
	{
		int w = index % width;
		int c = index / width;
		for (int h = height - 1; h > 0; h--) {
			int out_idx = (c * height + h) * width + w;
			int wei_idx = h * width + w;
			*(gradWeight + wei_idx) += *(gradData + out_idx)
					* (*(inter + out_idx));
			*(gradData + out_idx - width) += *(weight + wei_idx)
					* (*(gradData + out_idx));
			*(gradData + out_idx) *= 1 - *(weight + wei_idx);
		}
	}
}

template<typename Dtype>
__global__ void rightToLeft_backward_impro(const int num_kernels,
		Dtype* gradData, Dtype* weight, Dtype* inter, Dtype* gradWeight,
		const int height, const int width) {
	CUDA_KERNEL_LOOP(index, num_kernels)
	{
		int h = index % height;
		int c = index / height;
		for (int w = 0; w < width - 1; w++) {
			int out_idx = (c * height + h) * width + w;
			int wei_idx = h * width + w;
			*(gradWeight + wei_idx + 1) += *(gradData + out_idx)
					* (*(inter + out_idx));
			*(gradData + out_idx + 1) += *(weight + wei_idx + 1)
					* (*(gradData + out_idx));
			*(gradData + out_idx) *= 1 - *(weight + wei_idx + 1);
		}
	}
}

template<typename Dtype>
__global__ void leftToRight_backward_impro(const int num_kernels,
		Dtype* gradData, Dtype* weight, Dtype* inter, Dtype* gradWeight,
		const int height, const int width) {
	CUDA_KERNEL_LOOP(index, num_kernels)
	{
		int h = index % height;
		int c = index / height;
		for (int w = width - 1; w > 0; w--) {
			int out_idx = (c * height + h) * width + w;
			int wei_idx = h * width + w;
			*(gradWeight + wei_idx) += *(gradData + out_idx)
					* (*(inter + out_idx));
			*(gradData + out_idx - 1) += *(weight + wei_idx)
					* (*(gradData + out_idx));
			*(gradData + out_idx) *= 1 - *(weight + wei_idx);
		}
	}
}

template<typename Dtype>
__global__ void computeGradEdge_impro(const int num_kernels, Dtype* gradEdge,
		Dtype* weight, Dtype* gradWeight, const float sigma_k,
		const float div) {
	CUDA_KERNEL_LOOP(index, num_kernels)
	{
		float grad = sigma_k * div * *(weight + index) * *(gradWeight + index);
		*(gradEdge + index) += grad;
	}
}

template<typename Dtype>
void domainTransform_grad_impro(cudaStream_t stream, Dtype* edge,
		Dtype* gradEdge, Dtype* gradData, Dtype* weight, Dtype* inter,
		Dtype* gradWeight, const int plane, const int height, const int width,
		const int num_iter, const float sigma_range, const int sigma_spatial) {

	for (int m = num_iter - 1; m >= 0; m--) {
		Dtype* weight_m = weight + m * height * width;
		Dtype* gradWeight_m = gradWeight + m * height * width;
		Dtype* inter_m = inter + (m + 1) * height * width * plane * 4;

		int dimSize = 1024;
		int num_kernels = width * plane;
		int grid = (num_kernels + dimSize - 1) / dimSize;
		inter_m -= height * width * plane;
		bottomToTop_backward_impro<<<grid, dimSize, 0, stream>>>(num_kernels,
				gradData, weight_m, inter_m, gradWeight_m, height, width);
		inter_m -= height * width * plane;
		topToBottom_backward_impro<<<grid, dimSize, 0, stream>>>(num_kernels,
				gradData, weight_m, inter_m, gradWeight_m, height, width);

		dimSize = 1024;
		num_kernels = height * plane;
		grid = (num_kernels + dimSize - 1) / dimSize;
		inter_m -= height * width * plane;
		rightToLeft_backward_impro<<<grid, dimSize, 0, stream>>>(num_kernels,
				gradData, weight_m, inter_m, gradWeight_m, height, width);
		inter_m -= height * width * plane;
		leftToRight_backward_impro<<<grid, dimSize, 0, stream>>>(num_kernels,
				gradData, weight_m, inter_m, gradWeight_m, height, width);

		float sigma_k = sqrt(3 / (pow(4, num_iter) - 1)) * sigma_spatial
				* pow(2, num_iter - m - 1);
		sigma_k = -sqrt(2) / sigma_k;
		float div = sigma_spatial / sigma_range;
		dimSize = 1024;
		num_kernels = height * width;
		grid = (num_kernels + dimSize - 1) / dimSize;
		computeGradEdge_impro<<<grid, dimSize, 0, stream>>>(num_kernels,
				gradEdge, weight_m, gradWeight_m, sigma_k, div);
	}
}
