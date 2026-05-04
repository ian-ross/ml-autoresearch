import torch


def test_dev_torch_runtime_is_cpu_only() -> None:
    assert torch.__version__ == "2.5.1+cpu"
    assert torch.version.cuda is None
    assert not torch.cuda.is_available()
