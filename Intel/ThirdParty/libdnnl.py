import ctypes
from PuzzleLib.Intel.ThirdParty.finddnnl import findDNNL


_libdnnl = findDNNL()


class dnnlError(Exception):
	pass

class dnnlOutOfMemory(dnnlError):
	pass

class dnnlInvalidArguments(dnnlError):
	pass

class dnnlUnimplemented(dnnlError):
	pass

class dnnlIteratorEnds(dnnlError):
	pass

class dnnlRuntimeError(dnnlError):
	pass

class dnnlNotRequired(dnnlError):
	pass


dnnlExceptions = {
	1: dnnlOutOfMemory,
	2: dnnlInvalidArguments,
	3: dnnlUnimplemented,
	4: dnnlIteratorEnds,
	5: dnnlRuntimeError,
	6: dnnlNotRequired
}


class dnnl_version_t(ctypes.Structure):
	_fields_ = [
		("major", ctypes.c_int),
		("minor", ctypes.c_int),
		("patch", ctypes.c_int),
		("hash", ctypes.c_char_p)
	]


dnnl_data_type_t = {
	"dnnl_data_type_undef": 0,
	"dnnl_f16": 1,
	"dnnl_bf16": 2,
	"dnnl_f32": 3,
	"dnnl_s32": 4,
	"dnnl_s8": 5,
	"dnnl_u8": 6
}


dnnl_format_kind_t = {
	"dnnl_format_kind_undef": 0,
	"dnnl_format_kind_any": 1,
	"dnnl_blocked": 2,
	"dnnl_format_kind_wino": 3,
	"dnnl_format_kind_rnn_packed": 4
}


dnnl_format_tag_t = {
	"dnnl_format_tag_undef": 0,
	"dnnl_format_tag_any": 1,

	"dnnl_a": 2,
	"dnnl_ab": 3,
	"dnnl_abc": 4,
	"dnnl_abcd": 5,
	"dnnl_abcde": 6,
	"dnnl_abcdef": 7,

	"dnnl_abdec": 8,
	"dnnl_acb": 9,
	"dnnl_acbde": 10,
	"dnnl_acdb": 11,
	"dnnl_acdeb": 12,
	"dnnl_ba": 13,
	"dnnl_bac": 14,
	"dnnl_bacd": 15,
	"dnnl_bca": 16,
	"dnnl_bcda": 17,
	"dnnl_bcdea": 18,
	"dnnl_cba": 19,
	"dnnl_cdba": 20,
	"dnnl_cdeba": 21,
	"dnnl_decab": 22
}


dnnl_prop_kind_t = {
	"dnnl_prop_kind_undef": 0,
	"dnnl_forward_training": 64,
	"dnnl_forward_inference": 96,
	"dnnl_backward": 128,
	"dnnl_backward_data": 160,
	"dnnl_backward_weights": 192,
	"dnnl_backward_bias": 193
}


dnnl_primitive_kind_t = {
	"dnnl_undefined_primitive": 0,
	"dnnl_reorder": 1,
	"dnnl_shuffle": 2,
	"dnnl_concat": 3,
	"dnnl_sum": 4,
	"dnnl_convolution": 5,
	"dnnl_deconvolution": 6,
	"dnnl_eltwise": 7,
	"dnnl_softmax": 8,
	"dnnl_pooling": 9,
	"dnnl_lrn": 10,
	"dnnl_batch_normalization": 11,
	"dnnl_inner_product": 12,
	"dnnl_rnn": 13,
	"dnnl_gemm": 14
}


dnnl_alg_kind_t = {
	"dnnl_alg_kind_undef": 0x0,

	"dnnl_convolution_direct": 0x1,
	"dnnl_convolution_winograd": 0x2,
	"dnnl_convolution_auto": 0x3,
	"dnnl_deconvolution_direct": 0xa,
	"dnnl_deconvolution_winograd": 0xb,

	"dnnl_eltwise_relu": 0x1f,
	"dnnl_eltwise_tanh": 0x2f,
	"dnnl_eltwise_elu": 0x3f,
	"dnnl_eltwise_square": 0x4f,
	"dnnl_eltwise_abs": 0x5f,
	"dnnl_eltwise_sqrt": 0x6f,
	"dnnl_eltwise_linear": 0x7f,
	"dnnl_eltwise_bounded_relu": 0x8f,
	"dnnl_eltwise_soft_relu": 0x9f,
	"dnnl_eltwise_logistic": 0xaf,
	"dnnl_eltwise_exp": 0xbf,
	"dnnl_eltwise_gelu": 0xcf,

	"dnnl_pooling_max": 0x1ff,
	"dnnl_pooling_avg_include_padding": 0x2ff,
	"dnnl_pooling_avg_exclude_padding": 0x3ff,

	"dnnl_lrn_across_channels": 0xaff,
	"dnnl_lrn_within_channel": 0xbff,

	"dnnl_vanilla_rnn": 0x1fff,
	"dnnl_vanilla_lstm": 0x2fff,
	"dnnl_vanilla_gru": 0x3fff,
	"dnnl_lbr_gru": 0x4fff
}


dnnl_normalization_flags_t = {
	"dnnl_use_global_stats": 0x1,
	"dnnl_use_scaleshift": 0x2,
	"dnnl_fuse_bn_relu": 0x4
}


dnnl_MAX_NDIMS = 12

dnnl_dim_t = ctypes.c_int64
dnnl_dims_t = dnnl_dim_t * dnnl_MAX_NDIMS


class dnnl_blocking_desc_t(ctypes.Structure):
	_fields_ = [
		("strides", dnnl_dims_t),
		("inner_nblks", ctypes.c_int),
		("inner_blks", dnnl_dims_t),
		("inner_idxs", dnnl_dims_t)
	]


dnnl_wino_memory_format_t = {
	"dnnl_wino_undef": 0,
	"dnnl_wino_wei_aaOIoi": 1,
	"dnnl_wino_wei_aaOio": 2,
	"dnnl_wino_wei_aaOBiOo": 3,
	"dnnl_wino_wei_OBaaIBOIio": 4
}


class dnnl_wino_desc_t(ctypes.Structure):
	_fields_ = [
		("wino_format", ctypes.c_int),
		("r", ctypes.c_int),
		("alpha", ctypes.c_int),
		("ic", ctypes.c_int),
		("oc", ctypes.c_int),
		("ic_block", ctypes.c_int),
		("oc_block", ctypes.c_int),
		("ic2_block", ctypes.c_int),
		("oc2_block", ctypes.c_int),
		("adj_scale", ctypes.c_float),
		("size", ctypes.c_size_t)
	]


dnnl_rnn_packed_memory_format_t = {
	"dnnl_packed_format_undef": 0,
	"dnnl_ldigo_p": 1,
	"dnnl_ldgoi_p": 2
}


dnnl_RNN_MAX_N_PARTS = 4


class dnnl_rnn_packed_desc_t(ctypes.Structure):
	_fields_ = [
		("format", ctypes.c_int),
		("n_parts", ctypes.c_int),
		("n", ctypes.c_int),
		("ldb", ctypes.c_int),
		("parts", ctypes.c_int * dnnl_RNN_MAX_N_PARTS),
		("part_pack_size", ctypes.c_size_t * dnnl_RNN_MAX_N_PARTS),
		("pack_part", ctypes.c_uint * dnnl_RNN_MAX_N_PARTS),
		("offset_compensation", ctypes.c_size_t),
		("size", ctypes.c_size_t),
		("reserved", ctypes.c_char * 200)
	]


dnnl_memory_extra_flags_t = {
	"dnnl_memory_extra_flag_none": 0x0,
	"dnnl_memory_extra_flag_compensation_conv_s8s8": 0x1,
	"dnnl_memory_extra_flag_scale_adjust": 0x2
}


class dnnl_memory_extra_desc_t(ctypes.Structure):
	_fields_ = [
		("flags", ctypes.c_uint64),
		("compensation_mask", ctypes.c_int),
		("scale_adjust", ctypes.c_float),
		("reserved", ctypes.c_char * 64)
	]


class dnnl_memory_desc_t(ctypes.Structure):
	class _format_desc(ctypes.Union):
		_fields_ = [
			("blocking", dnnl_blocking_desc_t),
			("wino_desc", dnnl_wino_desc_t),
			("rnn_packed_desc", dnnl_rnn_packed_desc_t)
		]

	_fields_ = [
		("ndims", ctypes.c_int),
		("dims", dnnl_dims_t),
		("data_type", ctypes.c_int),
		("padded_dims", dnnl_dims_t),
		("padded_offsets", dnnl_dims_t),
		("offset0", dnnl_dim_t),
		("format_kind", ctypes.c_int),
		("format_desc", _format_desc),
		("extra", dnnl_memory_extra_desc_t)
	]


class dnnl_convolution_desc_t(ctypes.Structure):
	_fields_ = [
		("primitive_kind", ctypes.c_int),
		("prop_kind", ctypes.c_int),
		("alg_kind", ctypes.c_int),
		("src_desc", dnnl_memory_desc_t),
		("diff_src_desc", dnnl_memory_desc_t),
		("weights_desc", dnnl_memory_desc_t),
		("diff_weights_desc", dnnl_memory_desc_t),
		("bias_desc", dnnl_memory_desc_t),
		("diff_bias_desc", dnnl_memory_desc_t),
		("dst_desc", dnnl_memory_desc_t),
		("diff_dst_desc", dnnl_memory_desc_t),
		("strides", dnnl_dims_t),
		("dilates", dnnl_dims_t),
		("padding", dnnl_dims_t * 2),
		("accum_data_type", ctypes.c_int)
	]


dnnl_deconvolution_desc_t = dnnl_convolution_desc_t


class dnnl_softmax_desc_t(ctypes.Structure):
	_fields_ = [
		("primitive_kind", ctypes.c_int),
		("prop_kind", ctypes.c_int),
		("data_desc", dnnl_memory_desc_t),
		("diff_desc", dnnl_memory_desc_t),
		("softmax_axis", ctypes.c_int)
	]


class dnnl_pooling_desc_t(ctypes.Structure):
	_fields_ = [
		("primitive_kind", ctypes.c_int),
		("prop_kind", ctypes.c_int),
		("alg_kind", ctypes.c_int),
		("src_desc", dnnl_memory_desc_t),
		("diff_src_desc", dnnl_memory_desc_t),
		("dst_desc", dnnl_memory_desc_t),
		("diff_dst_desc", dnnl_memory_desc_t),
		("strides", dnnl_dims_t),
		("kernel", dnnl_dims_t),
		("padding", dnnl_dims_t * 2),
		("accum_data_type", ctypes.c_int)
	]


class dnnl_lrn_desc_t(ctypes.Structure):
	_fields_ = [
		("primitive_kind", ctypes.c_int),
		("prop_kind", ctypes.c_int),
		("alg_kind", ctypes.c_int),
		("data_desc", dnnl_memory_desc_t),
		("diff_data_desc", dnnl_memory_desc_t),
		("local_size", ctypes.c_int),
		("lrn_alpha", ctypes.c_float),
		("lrn_beta", ctypes.c_float),
		("lrn_k", ctypes.c_float)
	]


class dnnl_batch_normalization_desc_t(ctypes.Structure):
	_fields_ = [
		("primitive_kind", ctypes.c_int),
		("prop_kind", ctypes.c_int),
		("data_desc", dnnl_memory_desc_t),
		("diff_data_desc", dnnl_memory_desc_t),
		("data_scaleshift_desc", dnnl_memory_desc_t),
		("diff_data_scaleshift_desc", dnnl_memory_desc_t),
		("stat_desc", dnnl_memory_desc_t),
		("batch_norm_epsilon", ctypes.c_float),
		("flags", ctypes.c_uint)
	]


dnnl_rnn_flags_t = {
	"dnnl_rnn_flags_undef": 0x0
}


dnnl_rnn_direction_t = {
	"dnnl_unidirectional_left2right": 0,
	"dnnl_unidirectional_right2left": 1,
	"dnnl_bidirectional_concat": 2,
	"dnnl_bidirectional_sum": 3
}


class dnnl_rnn_desc_t(ctypes.Structure):
	_fields_ = [
		("primitive_kind", ctypes.c_int),
		("prop_kind", ctypes.c_int),
		("cell_kind", ctypes.c_int),
		("direction", ctypes.c_int),

		("src_layer_desc", dnnl_memory_desc_t),
		("src_iter_desc", dnnl_memory_desc_t),
		("src_iter_c_desc", dnnl_memory_desc_t),
		("weights_layer_desc", dnnl_memory_desc_t),
		("weights_iter_desc", dnnl_memory_desc_t),
		("bias_desc", dnnl_memory_desc_t),
		("dst_layer_desc", dnnl_memory_desc_t),
		("dst_iter_desc", dnnl_memory_desc_t),
		("src_iter_c_desc", dnnl_memory_desc_t),
		("placeholder_desc", dnnl_memory_desc_t),
		("placeholder2_desc", dnnl_memory_desc_t),

		("diff_src_layer_desc", dnnl_memory_desc_t),
		("diff_src_iter_desc", dnnl_memory_desc_t),
		("diff_src_iter_c_desc", dnnl_memory_desc_t),
		("diff_weights_layer_desc", dnnl_memory_desc_t),
		("diff_weights_iter_desc", dnnl_memory_desc_t),
		("diff_bias_desc", dnnl_memory_desc_t),
		("diff_dst_layer_desc", dnnl_memory_desc_t),
		("diff_dst_iter_desc", dnnl_memory_desc_t),
		("diff_dst_iter_c_desc", dnnl_memory_desc_t),
		("diff_placeholder_desc", dnnl_memory_desc_t),
		("diff_placeholder2_desc", dnnl_memory_desc_t),

		("flags", ctypes.c_uint),
		("activation_kind", ctypes.c_int),
		("alpha", ctypes.c_float),
		("beta", ctypes.c_float)
	]


dnnl_engine_kind_t = {
	"dnnl_any_engine": 0,
	"dnnl_cpu": 1,
	"dnnl_gpu": 2
}


dnnl_ARG = {
	"dnnl_ARG_SRC_0": 1,
	"dnnl_ARG_SRC_1": 2,
	"dnnl_ARG_SRC_2": 3,

	"dnnl_ARG_DST_0": 17,
	"dnnl_ARG_DST_1": 18,
	"dnnl_ARG_DST_2": 19,

	"dnnl_ARG_WEIGHTS_0": 33,
	"dnnl_ARG_WEIGHTS_1": 34,

	"dnnl_ARG_BIAS": 41,

	"dnnl_ARG_MEAN": 49,
	"dnnl_ARG_VARIANCE": 50,

	"dnnl_ARG_WORKSPACE": 64,
	"dnnl_ARG_SCRATCHPAD": 80,

	"dnnl_ARG_DIFF_SRC_0": 129,
	"dnnl_ARG_DIFF_SRC_1": 130,
	"dnnl_ARG_DIFF_SRC_2": 131,

	"dnnl_ARG_DIFF_DST_0": 145,
	"dnnl_ARG_DIFF_DST_1": 146,
	"dnnl_ARG_DIFF_DST_2": 147,

	"dnnl_ARG_DIFF_WEIGHTS_0": 161,
	"dnnl_ARG_DIFF_WEIGHTS_1": 162,

	"dnnl_ARG_DIFF_BIAS": 169,

	"dnnl_ARG_MULTIPLE_SRC": 1024,
	"dnnl_ARG_MULTIPLE_DST": 2048
}


class dnnl_exec_arg_t(ctypes.Structure):
	_fields_ = [
		("arg", ctypes.c_int),
		("memory", ctypes.c_void_p)
	]


dnnl_query_t = {
	"dnnl_query_undef": 0,

	"dnnl_query_engine": 1,
	"dnnl_query_primitive_kind": 2,

	"dnnl_query_num_of_inputs_s32": 3,
	"dnnl_query_num_of_outputs_s32": 4,

	"dnnl_query_time_estimate_f64": 5,
	"dnnl_query_memory_consumption_s64": 6,

	"dnnl_query_scratchpad_engine": 7,
	"dnnl_query_impl_info_str": 8,

	"dnnl_query_some_d": 64,
	"dnnl_query_op_d": 65,
	"dnnl_query_convolution_d": 66,
	"dnnl_query_deconvolution_d": 67,
	"dnnl_query_shuffle_d": 68,
	"dnnl_query_eltwise_d": 69,
	"dnnl_query_softmax_d": 70,
	"dnnl_query_pooling_d": 71,
	"dnnl_query_lrn_d": 72,
	"dnnl_query_batch_normalization_d": 73,
	"dnnl_query_inner_product_d": 74,
	"dnnl_query_rnn_d": 75,
	"dnnl_query_gemm_d": 76,

	"dnnl_query_some_md": 128,
	"dnnl_query_src_md": 129,
	"dnnl_query_diff_src_md": 130,
	"dnnl_query_weights_md": 131,
	"dnnl_query_diff_weights_md": 132,
	"dnnl_query_dst_md": 133,
	"dnnl_query_diff_dst_md": 134,
	"dnnl_query_workspace_md": 135,
	"dnnl_query_scratchpad_md": 136
}


dnnl_stream_flags_t = {
	"dnnl_stream_default_order": 0x1,
	"dnnl_stream_in_order": 0x2,
	"dnnl_stream_out_of_order": 0x4
}


def dnnlCheckStatus(status):
	if status != 0:
		try:
			raise dnnlExceptions[status]
		except KeyError:
			raise dnnlError


_libdnnl.dnnl_primitive_attr_create.restype = int
_libdnnl.dnnl_primitive_attr_create.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
def dnnl_primitive_attr_create():
	attr = ctypes.c_void_p()

	status = _libdnnl.dnnl_primitive_attr_create(ctypes.byref(attr))
	dnnlCheckStatus(status)

	return attr


_libdnnl.dnnl_primitive_attr_destroy.restype = int
_libdnnl.dnnl_primitive_attr_destroy.argtypes = [ctypes.c_void_p]
def dnnl_primitive_attr_destroy(attr):
	status = _libdnnl.dnnl_primitive_attr_destroy(attr)
	dnnlCheckStatus(status)


_libdnnl.dnnl_primitive_attr_get_output_scales.restype = int
_libdnnl.dnnl_primitive_attr_get_output_scales.argtypes = [
	ctypes.c_void_p, ctypes.POINTER(dnnl_dim_t), ctypes.POINTER(ctypes.c_int),
	ctypes.POINTER(ctypes.POINTER(ctypes.c_float))
]
def dnnl_primitive_attr_get_output_scales(attr):
	count, mask = dnnl_dim_t(), ctypes.c_int()
	scales = ctypes.POINTER(ctypes.c_float)()

	status = _libdnnl.dnnl_primitive_attr_get_output_scales(
		attr, ctypes.byref(count), ctypes.byref(mask), ctypes.byref(scales)
	)
	dnnlCheckStatus(status)

	return mask.value, list(scales[:count.value])


_libdnnl.dnnl_primitive_attr_set_output_scales.restype = int
_libdnnl.dnnl_primitive_attr_set_output_scales.argtypes = [
	ctypes.c_void_p, dnnl_dim_t, ctypes.c_int, ctypes.POINTER(ctypes.c_float)
]
def dnnl_primitive_attr_set_output_scales(attr, mask, scales):
	scales = (ctypes.c_float * len(scales))(*scales)

	status = _libdnnl.dnnl_primitive_attr_set_output_scales(attr, len(scales), mask, scales)
	dnnlCheckStatus(status)


_libdnnl.dnnl_primitive_desc_create.restype = int
_libdnnl.dnnl_primitive_desc_create.argtypes = [
	ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
]
def dnnl_primitive_desc_create(op_desc, attr, engine, hint_forward_primitive_desc):
	primitive_desc = ctypes.c_void_p()

	status = _libdnnl.dnnl_primitive_desc_create(
		ctypes.byref(primitive_desc), ctypes.addressof(op_desc), attr, engine, hint_forward_primitive_desc
	)
	dnnlCheckStatus(status)

	return primitive_desc.value


_libdnnl.dnnl_primitive_desc_destroy.restype = int
_libdnnl.dnnl_primitive_desc_destroy.argtypes = [ctypes.c_void_p]
def dnnl_primitive_desc_destroy(primitive_desc):
	status = _libdnnl.dnnl_primitive_desc_destroy(primitive_desc)
	dnnlCheckStatus(status)


_libdnnl.dnnl_primitive_desc_query_md.restype = ctypes.POINTER(dnnl_memory_desc_t)
_libdnnl.dnnl_primitive_desc_query_md.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
def dnnl_primitive_desc_query_md(primitive_desc, what, index):
	return _libdnnl.dnnl_primitive_desc_query_md(primitive_desc, what, index)


_libdnnl.dnnl_primitive_create.restype = int
_libdnnl.dnnl_primitive_create.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
def dnnl_primitive_create(primitive_desc):
	primitive = ctypes.c_void_p()

	status = _libdnnl.dnnl_primitive_create(ctypes.byref(primitive), primitive_desc)
	dnnlCheckStatus(status)

	return primitive.value


_libdnnl.dnnl_primitive_execute.restype = int
_libdnnl.dnnl_primitive_execute.argtypes = [
	ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(dnnl_exec_arg_t)
]
def dnnl_primitive_execute(primitive, stream, args):
	nargs = len(args)
	args = (dnnl_exec_arg_t * nargs)(*args)

	status = _libdnnl.dnnl_primitive_execute(primitive, stream, nargs, args)
	dnnlCheckStatus(status)


_libdnnl.dnnl_primitive_destroy.restype = int
_libdnnl.dnnl_primitive_destroy.argtypes = [ctypes.c_void_p]
def dnnl_primitive_destroy(primitive):
	status = _libdnnl.dnnl_primitive_destroy(primitive)
	dnnlCheckStatus(status)


_libdnnl.dnnl_memory_desc_init_by_tag.restype = int
_libdnnl.dnnl_memory_desc_init_by_tag.argtypes = [
	ctypes.POINTER(dnnl_memory_desc_t), ctypes.c_int, ctypes.POINTER(dnnl_dim_t), ctypes.c_int, ctypes.c_int
]
def dnnl_memory_desc_init_by_tag(dims, data_type, tag):
	memory_desc = dnnl_memory_desc_t()

	ndims = len(dims)
	dims = (dnnl_dim_t * len(dims))(*dims)

	status = _libdnnl.dnnl_memory_desc_init_by_tag(ctypes.byref(memory_desc), ndims, dims, data_type, tag)
	dnnlCheckStatus(status)

	return memory_desc


_libdnnl.dnnl_memory_desc_get_size.restype = ctypes.c_size_t
_libdnnl.dnnl_memory_desc_get_size.argtypes = [ctypes.POINTER(dnnl_memory_desc_t)]
def dnnl_memory_desc_get_size(memory_desc):
	if isinstance(memory_desc, dnnl_memory_desc_t):
		memory_desc = ctypes.byref(memory_desc)

	return _libdnnl.dnnl_memory_desc_get_size(memory_desc)


_libdnnl.dnnl_memory_create.restype = int
_libdnnl.dnnl_memory_create.argtypes = [
	ctypes.c_void_p, ctypes.POINTER(dnnl_memory_desc_t), ctypes.c_void_p, ctypes.c_void_p
]
def dnnl_memory_create(memory_desc, engine):
	memory = ctypes.c_void_p()

	if isinstance(memory_desc, dnnl_memory_desc_t):
		memory_desc = ctypes.byref(memory_desc)

	status = _libdnnl.dnnl_memory_create(ctypes.byref(memory), memory_desc, engine, None)
	dnnlCheckStatus(status)

	return memory


_libdnnl.dnnl_memory_get_data_handle.restype = int
_libdnnl.dnnl_memory_get_data_handle.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
def dnnl_memory_get_data_handle(memory):
	handle = ctypes.c_void_p()

	status = _libdnnl.dnnl_memory_get_data_handle(memory, ctypes.byref(handle))
	dnnlCheckStatus(status)

	return handle.value


_libdnnl.dnnl_memory_set_data_handle.restype = int
_libdnnl.dnnl_memory_set_data_handle.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
def dnnl_memory_set_data_handle(memory, handle):
	status = _libdnnl.dnnl_memory_set_data_handle(memory, handle)
	dnnlCheckStatus(status)


_libdnnl.dnnl_memory_destroy.restype = int
_libdnnl.dnnl_memory_destroy.argtypes = [ctypes.c_void_p]
def dnnl_memory_destroy(memory):
	status = _libdnnl.dnnl_memory_destroy(memory)
	dnnlCheckStatus(status)


_libdnnl.dnnl_reorder_primitive_desc_create.restype = int
_libdnnl.dnnl_reorder_primitive_desc_create.argtypes = [
	ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(dnnl_memory_desc_t), ctypes.c_void_p,
	ctypes.POINTER(dnnl_memory_desc_t), ctypes.c_void_p, ctypes.c_void_p
]
def dnnl_reorder_primitive_desc_create(src_md, src_engine, dst_md, dst_engine, attr):
	reorder_primitive_desc = ctypes.c_void_p()

	status = _libdnnl.dnnl_reorder_primitive_desc_create(
		ctypes.byref(reorder_primitive_desc), src_md, src_engine, dst_md, dst_engine, attr
	)
	dnnlCheckStatus(status)

	return reorder_primitive_desc.value


_libdnnl.dnnl_convolution_forward_desc_init.restype = int
_libdnnl.dnnl_convolution_forward_desc_init.argtypes = [
	ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.POINTER(dnnl_memory_desc_t),
	ctypes.POINTER(dnnl_memory_desc_t), ctypes.POINTER(dnnl_memory_desc_t), ctypes.POINTER(dnnl_memory_desc_t),
	ctypes.POINTER(dnnl_dim_t), ctypes.POINTER(dnnl_dim_t), ctypes.POINTER(dnnl_dim_t)
]
def dnnl_convolution_forward_desc_init(prop_kind, alg_kind, src_desc, weights_desc, bias_desc, dst_desc, strides,
										 padding):
	conv_desc = dnnl_convolution_desc_t()

	strides = (dnnl_dim_t * len(strides))(*strides)
	padding = (dnnl_dim_t * len(padding))(*padding)

	if bias_desc is not None:
		bias_desc = ctypes.byref(bias_desc)

	status = _libdnnl.dnnl_convolution_forward_desc_init(
		ctypes.byref(conv_desc), prop_kind, alg_kind, ctypes.byref(src_desc), ctypes.byref(weights_desc),
		bias_desc, ctypes.byref(dst_desc), strides, padding, None
	)
	dnnlCheckStatus(status)

	return conv_desc


_libdnnl.dnnl_dilated_convolution_forward_desc_init.restype = int
_libdnnl.dnnl_dilated_convolution_forward_desc_init.argtypes = [
	ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.POINTER(dnnl_memory_desc_t),
	ctypes.POINTER(dnnl_memory_desc_t), ctypes.POINTER(dnnl_memory_desc_t), ctypes.POINTER(dnnl_memory_desc_t),
	ctypes.POINTER(dnnl_dim_t), ctypes.POINTER(dnnl_dim_t), ctypes.POINTER(dnnl_dim_t),
	ctypes.POINTER(dnnl_dim_t)
]
def dnnl_dilated_convolution_forward_desc_init(prop_kind, alg_kind, src_desc, weights_desc, bias_desc, dst_desc,
												 strides, dilates, padding):
	conv_desc = dnnl_convolution_desc_t()

	strides = (dnnl_dim_t * len(strides))(*strides)
	dilates = (dnnl_dim_t * len(dilates))(*dilates)
	padding = (dnnl_dim_t * len(padding))(*padding)

	if bias_desc is not None:
		bias_desc = ctypes.byref(bias_desc)

	status = _libdnnl.dnnl_dilated_convolution_forward_desc_init(
		ctypes.byref(conv_desc), prop_kind, alg_kind, ctypes.byref(src_desc), ctypes.byref(weights_desc),
		bias_desc, ctypes.byref(dst_desc), strides, dilates, padding, None
	)
	dnnlCheckStatus(status)

	return conv_desc


_libdnnl.dnnl_convolution_backward_data_desc_init.restype = int
_libdnnl.dnnl_convolution_backward_data_desc_init.argtypes = [
	ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(dnnl_memory_desc_t), ctypes.POINTER(dnnl_memory_desc_t),
	ctypes.POINTER(dnnl_memory_desc_t), ctypes.POINTER(dnnl_dim_t), ctypes.POINTER(dnnl_dim_t),
	ctypes.POINTER(dnnl_dim_t)
]
def dnnl_convolution_backward_data_desc_init(alg_kind, diff_src_desc, weights_desc, diff_dst_desc, strides, padding):
	conv_desc = dnnl_convolution_desc_t()

	strides = (dnnl_dim_t * len(strides))(*strides)
	padding = (dnnl_dim_t * len(padding))(*padding)

	status = _libdnnl.dnnl_convolution_backward_data_desc_init(
		ctypes.byref(conv_desc), alg_kind, ctypes.byref(diff_src_desc), ctypes.byref(weights_desc),
		ctypes.byref(diff_dst_desc), strides, padding, None
	)
	dnnlCheckStatus(status)

	return conv_desc


_libdnnl.dnnl_dilated_convolution_backward_data_desc_init.restype = int
_libdnnl.dnnl_dilated_convolution_backward_data_desc_init.argtypes = [
	ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(dnnl_memory_desc_t), ctypes.POINTER(dnnl_memory_desc_t),
	ctypes.POINTER(dnnl_memory_desc_t), ctypes.POINTER(dnnl_dim_t), ctypes.POINTER(dnnl_dim_t),
	ctypes.POINTER(dnnl_dim_t), ctypes.POINTER(dnnl_dim_t)
]
def dnnl_dilated_convolution_backward_data_desc_init(alg_kind, diff_src_desc, weights_desc, diff_dst_desc, strides,
													   dilates, padding):
	conv_desc = dnnl_convolution_desc_t()

	strides = (dnnl_dim_t * len(strides))(*strides)
	dilates = (dnnl_dim_t * len(dilates))(*dilates)
	padding = (dnnl_dim_t * len(padding))(*padding)

	status = _libdnnl.dnnl_dilated_convolution_backward_data_desc_init(
		ctypes.byref(conv_desc), alg_kind, ctypes.byref(diff_src_desc), ctypes.byref(weights_desc),
		ctypes.byref(diff_dst_desc), strides, dilates, padding, None
	)
	dnnlCheckStatus(status)

	return conv_desc


_libdnnl.dnnl_convolution_backward_weights_desc_init.restype = int
_libdnnl.dnnl_convolution_backward_weights_desc_init.argtypes = [
	ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(dnnl_memory_desc_t), ctypes.POINTER(dnnl_memory_desc_t),
	ctypes.POINTER(dnnl_memory_desc_t), ctypes.POINTER(dnnl_memory_desc_t), ctypes.POINTER(dnnl_dim_t),
	ctypes.POINTER(dnnl_dim_t), ctypes.POINTER(dnnl_dim_t)
]
def dnnl_convolution_backward_weights_desc_init(alg_kind, src_desc, diff_weights_desc, diff_bias_desc, diff_dst_desc,
												  strides, padding):
	conv_desc = dnnl_convolution_desc_t()

	strides = (dnnl_dim_t * len(strides))(*strides)
	padding = (dnnl_dim_t * len(padding))(*padding)

	if diff_bias_desc is not None:
		diff_bias_desc = ctypes.byref(diff_bias_desc)

	status = _libdnnl.dnnl_convolution_backward_weights_desc_init(
		ctypes.byref(conv_desc), alg_kind, ctypes.byref(src_desc), ctypes.byref(diff_weights_desc),
		diff_bias_desc, ctypes.byref(diff_dst_desc), strides, padding, None
	)
	dnnlCheckStatus(status)

	return conv_desc


_libdnnl.dnnl_dilated_convolution_backward_weights_desc_init.restype = int
_libdnnl.dnnl_dilated_convolution_backward_weights_desc_init.argtypes = [
	ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(dnnl_memory_desc_t), ctypes.POINTER(dnnl_memory_desc_t),
	ctypes.POINTER(dnnl_memory_desc_t), ctypes.POINTER(dnnl_memory_desc_t), ctypes.POINTER(dnnl_dim_t),
	ctypes.POINTER(dnnl_dim_t), ctypes.POINTER(dnnl_dim_t), ctypes.POINTER(dnnl_dim_t)
]
def dnnl_dilated_convolution_backward_weights_desc_init(alg_kind, src_desc, diff_weights_desc, diff_bias_desc,
														  diff_dst_desc, strides, dilates, padding):
	conv_desc = dnnl_convolution_desc_t()

	strides = (dnnl_dim_t * len(strides))(*strides)
	dilates = (dnnl_dim_t * len(dilates))(*dilates)
	padding = (dnnl_dim_t * len(padding))(*padding)

	status = _libdnnl.dnnl_dilated_convolution_backward_weights_desc_init(
		ctypes.byref(conv_desc), alg_kind, ctypes.byref(src_desc), ctypes.byref(diff_weights_desc),
		ctypes.byref(diff_bias_desc), ctypes.byref(diff_dst_desc), strides, dilates, padding, None
	)
	dnnlCheckStatus(status)

	return conv_desc


_libdnnl.dnnl_deconvolution_forward_desc_init.restype = int
_libdnnl.dnnl_deconvolution_forward_desc_init.argtypes = [
	ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.POINTER(dnnl_memory_desc_t),
	ctypes.POINTER(dnnl_memory_desc_t), ctypes.POINTER(dnnl_memory_desc_t), ctypes.POINTER(dnnl_memory_desc_t),
	ctypes.POINTER(dnnl_dim_t), ctypes.POINTER(dnnl_dim_t), ctypes.POINTER(dnnl_dim_t)
]
def dnnl_deconvolution_forward_desc_init(prop_kind, alg_kind, src_desc, weights_desc, bias_desc, dst_desc, strides,
										   padding):
	conv_desc = dnnl_deconvolution_desc_t()

	strides = (dnnl_dim_t * len(strides))(*strides)
	padding = (dnnl_dim_t * len(padding))(*padding)

	if bias_desc is not None:
		bias_desc = ctypes.byref(bias_desc)

	status = _libdnnl.dnnl_deconvolution_forward_desc_init(
		ctypes.byref(conv_desc), prop_kind, alg_kind, ctypes.byref(src_desc), ctypes.byref(weights_desc),
		bias_desc, ctypes.byref(dst_desc), strides, padding, None
	)
	dnnlCheckStatus(status)

	return conv_desc


_libdnnl.dnnl_dilated_deconvolution_forward_desc_init.restype = int
_libdnnl.dnnl_dilated_deconvolution_forward_desc_init.argtypes = [
	ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.POINTER(dnnl_memory_desc_t),
	ctypes.POINTER(dnnl_memory_desc_t), ctypes.POINTER(dnnl_memory_desc_t), ctypes.POINTER(dnnl_memory_desc_t),
	ctypes.POINTER(dnnl_dim_t), ctypes.POINTER(dnnl_dim_t), ctypes.POINTER(dnnl_dim_t),
	ctypes.POINTER(dnnl_dim_t)
]
def dnnl_dilated_deconvolution_forward_desc_init(prop_kind, alg_kind, src_desc, weights_desc, bias_desc, dst_desc,
												   strides, dilates, padding):
	conv_desc = dnnl_deconvolution_desc_t()

	strides = (dnnl_dim_t * len(strides))(*strides)
	dilates = (dnnl_dim_t * len(dilates))(*dilates)
	padding = (dnnl_dim_t * len(padding))(*padding)

	if bias_desc is not None:
		bias_desc = ctypes.byref(bias_desc)

	status = _libdnnl.dnnl_dilated_deconvolution_forward_desc_init(
		ctypes.byref(conv_desc), prop_kind, alg_kind, ctypes.byref(src_desc), ctypes.byref(weights_desc),
		bias_desc, ctypes.byref(dst_desc), strides, dilates, padding, None
	)
	dnnlCheckStatus(status)

	return conv_desc


_libdnnl.dnnl_deconvolution_backward_data_desc_init.restype = int
_libdnnl.dnnl_deconvolution_backward_data_desc_init.argtypes = [
	ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(dnnl_memory_desc_t), ctypes.POINTER(dnnl_memory_desc_t),
	ctypes.POINTER(dnnl_memory_desc_t), ctypes.POINTER(dnnl_dim_t), ctypes.POINTER(dnnl_dim_t),
	ctypes.POINTER(dnnl_dim_t)
]
def dnnl_deconvolution_backward_data_desc_init(alg_kind, diff_src_desc, weights_desc, diff_dst_desc, strides,
												 padding):
	conv_desc = dnnl_deconvolution_desc_t()

	strides = (dnnl_dim_t * len(strides))(*strides)
	padding = (dnnl_dim_t * len(padding))(*padding)

	status = _libdnnl.dnnl_deconvolution_backward_data_desc_init(
		ctypes.byref(conv_desc), alg_kind, ctypes.byref(diff_src_desc), ctypes.byref(weights_desc),
		ctypes.byref(diff_dst_desc), strides, padding, None
	)
	dnnlCheckStatus(status)

	return conv_desc


_libdnnl.dnnl_dilated_deconvolution_backward_data_desc_init.restype = int
_libdnnl.dnnl_dilated_deconvolution_backward_data_desc_init.argtypes = [
	ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(dnnl_memory_desc_t), ctypes.POINTER(dnnl_memory_desc_t),
	ctypes.POINTER(dnnl_memory_desc_t), ctypes.POINTER(dnnl_dim_t), ctypes.POINTER(dnnl_dim_t),
	ctypes.POINTER(dnnl_dim_t), ctypes.POINTER(dnnl_dim_t)
]
def dnnl_dilated_deconvolution_backward_data_desc_init(alg_kind, diff_src_desc, weights_desc, diff_dst_desc, strides,
														 dilates, padding):
	conv_desc = dnnl_deconvolution_desc_t()

	strides = (dnnl_dim_t * len(strides))(*strides)
	dilates = (dnnl_dim_t * len(dilates))(*dilates)
	padding = (dnnl_dim_t * len(padding))(*padding)

	status = _libdnnl.dnnl_dilated_deconvolution_backward_data_desc_init(
		ctypes.byref(conv_desc), alg_kind, ctypes.byref(diff_src_desc), ctypes.byref(weights_desc),
		ctypes.byref(diff_dst_desc), strides, dilates, padding, None
	)
	dnnlCheckStatus(status)

	return conv_desc


_libdnnl.dnnl_deconvolution_backward_weights_desc_init.restype = int
_libdnnl.dnnl_deconvolution_backward_weights_desc_init.argtypes = [
	ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(dnnl_memory_desc_t), ctypes.POINTER(dnnl_memory_desc_t),
	ctypes.POINTER(dnnl_memory_desc_t), ctypes.POINTER(dnnl_memory_desc_t), ctypes.POINTER(dnnl_dim_t),
	ctypes.POINTER(dnnl_dim_t), ctypes.POINTER(dnnl_dim_t)
]
def dnnl_deconvolution_backward_weights_desc_init(alg_kind, src_desc, diff_weights_desc, diff_bias_desc,
													diff_dst_desc, strides, padding):
	conv_desc = dnnl_deconvolution_desc_t()

	strides = (dnnl_dim_t * len(strides))(*strides)
	padding = (dnnl_dim_t * len(padding))(*padding)

	if diff_bias_desc is not None:
		diff_bias_desc = ctypes.byref(diff_bias_desc)

	status = _libdnnl.dnnl_deconvolution_backward_weights_desc_init(
		ctypes.byref(conv_desc), alg_kind, ctypes.byref(src_desc), ctypes.byref(diff_weights_desc),
		diff_bias_desc, ctypes.byref(diff_dst_desc), strides, padding, None
	)
	dnnlCheckStatus(status)

	return conv_desc


_libdnnl.dnnl_dilated_deconvolution_backward_weights_desc_init.restype = int
_libdnnl.dnnl_dilated_deconvolution_backward_weights_desc_init.argtypes = [
	ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(dnnl_memory_desc_t), ctypes.POINTER(dnnl_memory_desc_t),
	ctypes.POINTER(dnnl_memory_desc_t), ctypes.POINTER(dnnl_memory_desc_t), ctypes.POINTER(dnnl_dim_t),
	ctypes.POINTER(dnnl_dim_t), ctypes.POINTER(dnnl_dim_t), ctypes.POINTER(dnnl_dim_t)
]
def dnnl_dilated_deconvolution_backward_weights_desc_init(alg_kind, src_desc, diff_weights_desc, diff_bias_desc,
															diff_dst_desc, strides, dilates, padding):
	conv_desc = dnnl_deconvolution_desc_t()

	strides = (dnnl_dim_t * len(strides))(*strides)
	dilates = (dnnl_dim_t * len(dilates))(*dilates)
	padding = (dnnl_dim_t * len(padding))(*padding)

	if diff_bias_desc is not None:
		diff_bias_desc = ctypes.byref(diff_bias_desc)

	status = _libdnnl.dnnl_dilated_deconvolution_backward_weights_desc_init(
		ctypes.byref(conv_desc), alg_kind, ctypes.byref(src_desc), ctypes.byref(diff_weights_desc),
		diff_bias_desc, ctypes.byref(diff_dst_desc), strides, dilates, padding, None
	)
	dnnlCheckStatus(status)

	return conv_desc


_libdnnl.dnnl_softmax_forward_desc_init.restype = int
_libdnnl.dnnl_softmax_forward_desc_init.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
def dnnl_softmax_forward_desc_init(prop_kind, data_desc, softmax_axis):
	softmax_desc = dnnl_softmax_desc_t()

	status = _libdnnl.dnnl_softmax_forward_desc_init(
		ctypes.byref(softmax_desc), prop_kind, ctypes.byref(data_desc), softmax_axis
	)
	dnnlCheckStatus(status)

	return softmax_desc


_libdnnl.dnnl_softmax_backward_desc_init.restype = int
_libdnnl.dnnl_softmax_backward_desc_init.argtypes = [
	ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int
]
def dnnl_softmax_backward_desc_init(diff_desc, data_desc, softmax_axis):
	softmax_desc = dnnl_softmax_desc_t()

	status = _libdnnl.dnnl_softmax_backward_desc_init(
		ctypes.byref(softmax_desc), ctypes.byref(diff_desc), ctypes.byref(data_desc), softmax_axis
	)
	dnnlCheckStatus(status)

	return softmax_desc


_libdnnl.dnnl_pooling_forward_desc_init.restype = int
_libdnnl.dnnl_pooling_forward_desc_init.argtypes = [
	ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.POINTER(dnnl_memory_desc_t),
	ctypes.POINTER(dnnl_memory_desc_t), ctypes.POINTER(dnnl_dim_t), ctypes.POINTER(dnnl_dim_t),
	ctypes.POINTER(dnnl_dim_t), ctypes.POINTER(dnnl_dim_t)
]
def dnnl_pooling_forward_desc_init(prop_kind, alg_kind, src_desc, dst_desc, strides, kernel, padding):
	pool_desc = dnnl_pooling_desc_t()

	strides = (dnnl_dim_t * len(strides))(*strides)
	kernel = (dnnl_dim_t * len(kernel))(*kernel)
	padding = (dnnl_dim_t * len(padding))(*padding)

	status = _libdnnl.dnnl_pooling_forward_desc_init(
		ctypes.byref(pool_desc), prop_kind, alg_kind, ctypes.byref(src_desc), ctypes.byref(dst_desc),
		strides, kernel, padding, None
	)
	dnnlCheckStatus(status)

	return pool_desc


_libdnnl.dnnl_pooling_backward_desc_init.restype = int
_libdnnl.dnnl_pooling_backward_desc_init.argtypes = [
	ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(dnnl_memory_desc_t), ctypes.POINTER(dnnl_memory_desc_t),
	ctypes.POINTER(dnnl_dim_t), ctypes.POINTER(dnnl_dim_t), ctypes.POINTER(dnnl_dim_t),
	ctypes.POINTER(dnnl_dim_t)
]
def dnnl_pooling_backward_desc_init(alg_kind, diff_src_desc, diff_dst_desc, strides, kernel, padding):
	pool_desc = dnnl_pooling_desc_t()

	strides = (dnnl_dim_t * len(strides))(*strides)
	kernel = (dnnl_dim_t * len(kernel))(*kernel)
	padding = (dnnl_dim_t * len(padding))(*padding)

	status = _libdnnl.dnnl_pooling_backward_desc_init(
		ctypes.byref(pool_desc), alg_kind, ctypes.byref(diff_src_desc), ctypes.byref(diff_dst_desc),
		strides, kernel, padding, None
	)
	dnnlCheckStatus(status)

	return pool_desc


_libdnnl.dnnl_lrn_forward_desc_init.restype = int
_libdnnl.dnnl_lrn_forward_desc_init.argtypes = [
	ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.POINTER(dnnl_memory_desc_t),
	dnnl_dim_t, ctypes.c_float, ctypes.c_float, ctypes.c_float
]
def dnnl_lrn_forward_desc_init(prop_kind, alg_kind, data_desc, local_size, alpha, beta, k):
	lrn_desc = dnnl_lrn_desc_t()

	status = _libdnnl.dnnl_lrn_forward_desc_init(
		ctypes.byref(lrn_desc), prop_kind, alg_kind, ctypes.byref(data_desc), local_size, alpha, beta, k
	)
	dnnlCheckStatus(status)

	return lrn_desc


_libdnnl.dnnl_lrn_backward_desc_init.restype = int
_libdnnl.dnnl_lrn_backward_desc_init.argtypes = [
	ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(dnnl_memory_desc_t), ctypes.POINTER(dnnl_memory_desc_t),
	dnnl_dim_t, ctypes.c_float, ctypes.c_float, ctypes.c_float
]
def dnnl_lrn_backward_desc_init(alg_kind, data_desc, diff_data_desc, local_size, alpha, beta, k):
	lrn_desc = dnnl_lrn_desc_t()

	status = _libdnnl.dnnl_lrn_backward_desc_init(
		ctypes.byref(lrn_desc), alg_kind, ctypes.byref(data_desc), ctypes.byref(diff_data_desc),
		local_size, alpha, beta, k
	)
	dnnlCheckStatus(status)

	return lrn_desc


_libdnnl.dnnl_batch_normalization_forward_desc_init.restype = int
_libdnnl.dnnl_batch_normalization_forward_desc_init.argtypes = [
	ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(dnnl_memory_desc_t), ctypes.c_float, ctypes.c_uint
]
def dnnl_batch_normalization_forward_desc_init(prop_kind, data_desc, epsilon, flags):
	bnrm_desc = dnnl_batch_normalization_desc_t()

	status = _libdnnl.dnnl_batch_normalization_forward_desc_init(
		ctypes.byref(bnrm_desc), prop_kind, ctypes.byref(data_desc), epsilon, flags
	)
	dnnlCheckStatus(status)

	return bnrm_desc


_libdnnl.dnnl_batch_normalization_backward_desc_init.restype = int
_libdnnl.dnnl_batch_normalization_backward_desc_init.argtypes = [
	ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(dnnl_memory_desc_t), ctypes.POINTER(dnnl_memory_desc_t),
	ctypes.c_float, ctypes.c_uint
]
def dnnl_batch_normalization_backward_desc_init(prop_kind, diff_data_desc, data_desc, epsilon, flags):
	bnrm_desc = dnnl_batch_normalization_desc_t()

	status = _libdnnl.dnnl_batch_normalization_backward_desc_init(
		ctypes.byref(bnrm_desc), prop_kind, ctypes.byref(diff_data_desc), ctypes.byref(data_desc), epsilon, flags
	)
	dnnlCheckStatus(status)

	return bnrm_desc


_libdnnl.dnnl_engine_get_count.restype = ctypes.c_size_t
_libdnnl.dnnl_engine_get_count.argtypes = [ctypes.c_int]
def dnnl_engine_get_count(kind):
	return _libdnnl.dnnl_engine_get_count(kind)


_libdnnl.dnnl_engine_create.restype = int
_libdnnl.dnnl_engine_create.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
def dnnl_engine_create(kind, index):
	engine = ctypes.c_void_p()

	status = _libdnnl.dnnl_engine_create(ctypes.byref(engine), kind, index)
	dnnlCheckStatus(status)

	return engine.value


_libdnnl.dnnl_engine_get_kind.restype = int
_libdnnl.dnnl_engine_get_kind.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
def dnnl_engine_get_kind(engine):
	kind = ctypes.c_int()

	status = _libdnnl.dnnl_engine_get_kind(engine, ctypes.byref(kind))
	dnnlCheckStatus(status)

	return kind.value


_libdnnl.dnnl_engine_destroy.restype = int
_libdnnl.dnnl_engine_destroy.argtypes = [ctypes.c_void_p]
def dnnl_engine_destroy(engine):
	status = _libdnnl.dnnl_engine_destroy(engine)
	dnnlCheckStatus(status)


_libdnnl.dnnl_stream_create.restype = int
_libdnnl.dnnl_stream_create.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
def dnnl_stream_create(engine, flags):
	stream = ctypes.c_void_p()

	status = _libdnnl.dnnl_stream_create(ctypes.byref(stream), engine, flags)
	dnnlCheckStatus(status)

	return stream.value


_libdnnl.dnnl_stream_wait.restype = int
_libdnnl.dnnl_stream_wait.argtypes = [ctypes.c_void_p]
def dnnl_stream_wait(stream):
	status = _libdnnl.dnnl_stream_wait(stream)
	dnnlCheckStatus(status)


_libdnnl.dnnl_stream_destroy.restype = int
_libdnnl.dnnl_stream_destroy.argtypes = [ctypes.c_void_p]
def dnnl_stream_destroy(stream):
	status = _libdnnl.dnnl_stream_destroy(stream)
	dnnlCheckStatus(status)


_libdnnl.dnnl_set_verbose.restype = int
_libdnnl.dnnl_set_verbose.argtypes = [ctypes.c_int]
def dnnl_verbose_set(level):
	status = _libdnnl.dnnl_set_verbose(level)
	dnnlCheckStatus(status)


_libdnnl.dnnl_version.restype = ctypes.POINTER(dnnl_version_t)
_libdnnl.dnnl_version.argtypes = []
def dnnl_version():
	version = _libdnnl.dnnl_version()
	return version.contents


_libdnnl.dnnl_sgemm.restype = int
_libdnnl.dnnl_sgemm.argtypes = [
	ctypes.c_char, ctypes.c_char, dnnl_dim_t, dnnl_dim_t, dnnl_dim_t, ctypes.c_float, ctypes.c_void_p,
	dnnl_dim_t, ctypes.c_void_p, dnnl_dim_t, ctypes.c_float, ctypes.c_void_p, dnnl_dim_t
]
def dnnl_sgemm(transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc):
	status = _libdnnl.dnnl_sgemm(ord(transA), ord(transB), M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)
	dnnlCheckStatus(status)
