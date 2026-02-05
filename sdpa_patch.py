import torch  # noqa: F401

torch.backends.cuda.enable_cudnn_sdp(True)
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cudnn.deterministic = False

_orig_sdpa = torch.nn.functional.scaled_dot_product_attention


def _sdpa_with_priority(*args, **kwargs):
    with torch.nn.attention.sdpa_kernel(
        [
            torch.nn.attention.SDPBackend.CUDNN_ATTENTION,
            torch.nn.attention.SDPBackend.FLASH_ATTENTION,
            torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
            torch.nn.attention.SDPBackend.MATH,
        ],
        set_priority=True,
    ):
        return _orig_sdpa(*args, **kwargs)


# 覆盖全局函数
torch.nn.functional.scaled_dot_product_attention = _sdpa_with_priority

print("SDPA patch applied")
