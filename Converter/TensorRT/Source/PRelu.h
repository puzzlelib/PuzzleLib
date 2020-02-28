#pragma once

#include <vector>
#include "Plugins.h"


struct PRelu : Plugin
{
	std::vector<float> m_scales;

	float *m_gpuscales = nullptr;
	size_t m_gpulength = 0;


	PRelu(const float *scales, size_t length);
	PRelu(const void *serialData, size_t serialLength);

	size_t getSerializationSize() override;
	void serialize(void *serialData) override;

	int enqueue(int batchSize, const void * const *inputs, void **outputs, void * /*workspace*/,
				cudaStream_t stream) override;

	nv::Dims getOutputDimensions(int /*index*/, const nv::Dims *inputDims, int /*nbInputs*/) override;

	void configureWithFormat(const nv::Dims *inputDims, int /*nbInputs*/, const nv::Dims *outputDims, int /*nbOutputs*/,
							 nv::DataType type, nv::PluginFormat format, int /*maxBatchSize*/) override;
	bool supportsFormat(nv::DataType type, nv::PluginFormat format) const override;

	int initialize() override;
	void terminate() override;

	size_t getWorkspaceSize(int /*maxBatchSize*/) const override;
	int getNbOutputs() const override;
};
