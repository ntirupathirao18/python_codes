import torch

def calculate_metrics(pred, gt, value):
    # True Positives (TP): pred value and gt value are both equal to the specified value
    TP = torch.sum(torch.logical_and(pred == value, gt == value))

    # False Positives (FP): pred value is equal to the specified value while gt value is not
    FP = torch.sum(torch.logical_and(pred == value, gt != value))

    # False Negatives (FN): pred value is not equal to the specified value while gt value is
    FN = torch.sum(torch.logical_and(pred != value, gt == value))

    return TP.item(), FP.item(), FN.item()

# Assuming pred and gt are PyTorch tensors of shape (1080, 1920)
# Replace pred_tensor and gt_tensor with your actual data
pred_tensor = torch.randint(0, 11, size=(1080, 1920))  # Example random prediction tensor
gt_tensor = torch.randint(0, 11, size=(1080, 1920))    # Example random ground truth tensor

# Generate values from 0 to 10 as a PyTorch tensor
values = torch.arange(11)

# Expand dimensions of pred_tensor and gt_tensor to perform element-wise comparison with values
pred_expanded = pred_tensor.unsqueeze(-1)
gt_expanded = gt_tensor.unsqueeze(-1)

# Compute TP, FP, FN for each value using tensor operations
TP = torch.sum(torch.logical_and(pred_expanded == values, gt_expanded == values), dim=(0, 1))
FP = torch.sum(torch.logical_and(pred_expanded == values, gt_expanded != values), dim=(0, 1))
FN = torch.sum(torch.logical_and(pred_expanded != values, gt_expanded == values), dim=(0, 1))

print("True Positives (TP):", TP.tolist())
print("False Positives (FP):", FP.tolist())
print("False Negatives (FN):", FN.tolist())



