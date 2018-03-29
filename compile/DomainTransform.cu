#include "THCUNN.h"
#include "common.h"
#include "cuda_dt_improved.h"

void THNN_CudaDomainTransform_updateOutput(THCState *state, THCudaTensor *edge,
		THCudaTensor *output, THCudaTensor *weight, THCudaTensor *inter,
		int num_iter, float sigma_range, int sigma_spatial) {

	long batchSize = output->size[0];
	long plane = output->size[1];
	long height = output->size[2];
	long width = output->size[3];

	// Resize output
	THCudaTensor_resize4d(state, weight, batchSize, num_iter, height, width);
	THCudaTensor_resize4d(state, inter, batchSize, num_iter * plane * 4, height,
			width);
	THCudaTensor_fill(state, weight, -1);
	THCudaTensor_fill(state, inter, 0);

	THCudaTensor *edge_n = THCudaTensor_new(state);
	THCudaTensor *weight_n = THCudaTensor_new(state);
	THCudaTensor *output_n = THCudaTensor_new(state);
	THCudaTensor *inter_n = THCudaTensor_new(state);

	// For each elt in batch, do:
	for (int elt = 0; elt < batchSize; elt++) {
		// Matrix mulitply per output:
		THCudaTensor_select(state, edge_n, edge, 0, elt);
		THCudaTensor_select(state, weight_n, weight, 0, elt);
		THCudaTensor_select(state, output_n, output, 0, elt);
		THCudaTensor_select(state, inter_n, inter, 0, elt);

		domainTransform_impro(THCState_getCurrentStream(state),
				THCudaTensor_data(state, edge_n),
				THCudaTensor_data(state, weight_n),
				THCudaTensor_data(state, output_n),
				THCudaTensor_data(state, inter_n), plane, height, width,
				num_iter, sigma_range, sigma_spatial);
	}

	// Free
	THCudaTensor_free(state, edge_n);
	THCudaTensor_free(state, weight_n);
	THCudaTensor_free(state, output_n);
	THCudaTensor_free(state, inter_n);
}

void THNN_CudaDomainTransform_updateGradInput(THCState *state,
		THCudaTensor *edge, THCudaTensor *gradData, THCudaTensor *gradEdge,
		THCudaTensor *weight, THCudaTensor *inter, THCudaTensor *gradWeight,
		int num_iter, float sigma_range, int sigma_spatial) {

	long batchSize = gradData->size[0];
	long plane = gradData->size[1];
	long height = gradData->size[2];
	long width = gradData->size[3];

	THCudaTensor *edge_n = THCudaTensor_new(state);
	THCudaTensor *gradEdge_n = THCudaTensor_new(state);
	THCudaTensor *gradData_n = THCudaTensor_new(state);
	THCudaTensor *weight_n = THCudaTensor_new(state);
	THCudaTensor *inter_n = THCudaTensor_new(state);
	THCudaTensor *gradWeight_n = THCudaTensor_new(state);

	for (int elt = 0; elt < batchSize; elt++) {
		THCudaTensor_select(state, edge_n, edge, 0, elt);
		THCudaTensor_select(state, gradEdge_n, gradEdge, 0, elt);
		THCudaTensor_select(state, gradData_n, gradData, 0, elt);
		THCudaTensor_select(state, weight_n, weight, 0, elt);
		THCudaTensor_select(state, inter_n, inter, 0, elt);
		THCudaTensor_select(state, gradWeight_n, gradWeight, 0, elt);

		domainTransform_grad_impro(THCState_getCurrentStream(state),
				THCudaTensor_data(state, edge_n),
				THCudaTensor_data(state, gradEdge_n),
				THCudaTensor_data(state, gradData_n),
				THCudaTensor_data(state, weight_n),
				THCudaTensor_data(state, inter_n),
				THCudaTensor_data(state, gradWeight_n), plane, height, width,
				num_iter, sigma_range, sigma_spatial);
	}

	THCudaTensor_free(state, edge_n);
	THCudaTensor_free(state, gradEdge_n);
	THCudaTensor_free(state, gradData_n);
	THCudaTensor_free(state, weight_n);
	THCudaTensor_free(state, inter_n);
	THCudaTensor_free(state, gradWeight_n);
}
