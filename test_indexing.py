import torch

B, C, N = 2, 5, 10
M = 3
pd_scores = torch.randn(B, C, N)
gt_labels_ind = torch.randint(0, C, (B, M))
batch_ind = torch.arange(B).view(-1, 1).expand(-1, M)

result = pd_scores[batch_ind, gt_labels_ind]
print(f"Result shape: {result.shape}") # Should be (B, M, N)
