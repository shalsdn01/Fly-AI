import torch
from transformers import AutoModel

# parameter check
model = AutoModel.from_pretrained("skt/kobert-base-v1")

state_dict = model.state_dict()

total_params = 0
total_bytes = 0

print(f"{'Name':80s} {'Shape':40s} {'Dtype':20s} {'Bytes'}")
print("-" * 140)

for name, param in state_dict.items():
    numel = param.numel()
    dtype = param.dtype
    bytes_per_elem = torch.finfo(dtype).bits // 8 if dtype.is_floating_point else torch.iinfo(dtype).bits // 8
    size_bytes = numel * bytes_per_elem

    total_params += numel
    total_bytes += size_bytes

    print(f"{name:80s} {str(tuple(param.shape)):40s} {str(dtype):20s} {size_bytes}")

print("-" * 140)
print(f"Total Parameters: {total_params:,}")
print(f"Estimated Memory Size: {total_bytes / (1024**2):.2f} MB")


""" =========================================================== """


# half parameters
model = model.half()

# 실제 모델의 파라미터를 담고있는 state_dict 값 분석
state_dict = model.state_dict()


total_params = 0
total_bytes = 0

print(f"{'Name':80s} {'Shape':40s} {'Dtype':20s} {'Bytes'}")
print("-" * 140)

for name, param in state_dict.items():
    numel = param.numel()
    dtype = param.dtype
    bytes_per_elem = torch.finfo(dtype).bits // 8 if dtype.is_floating_point else torch.iinfo(dtype).bits // 8
    size_bytes = numel * bytes_per_elem

    total_params += numel
    total_bytes += size_bytes

    print(f"{name:80s} {str(tuple(param.shape)):40s} {str(dtype):20s} {size_bytes}")

print("-" * 140)
print(f"Total Parameters: {total_params:,}")
print(f"Estimated Memory Size: {total_bytes / (1024 ** 2):.2f} MB")

torch.save(model.state_dict(), "fp16_kobert.pt")