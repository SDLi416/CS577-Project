import torch
import torch.nn as nn
from typing import Tuple

# https://github.com/JakubTomaszewski/Knowledge-Distillation-in-Semantic-Segmentation/blob/main/src/knowledge_distillation/losses.py#L37


# ce_loss = nn.CrossEntropyLoss()
# mse_loss = nn.MSELoss()
# inputs, labels = inputs.to(device), labels.to(device)

# optimizer.zero_grad()

# # Again ignore teacher logits
# with torch.no_grad():
#     _, teacher_feature_map = teacher(inputs)

# # Forward pass with the student model
# student_logits, regressor_feature_map = student(inputs)

# # Calculate the loss
# hidden_rep_loss = mse_loss(regressor_feature_map, teacher_feature_map)

# # Calculate the true label loss
# label_loss = ce_loss(student_logits, labels)

# # Weighted sum of the two losses
# loss = feature_map_weight * hidden_rep_loss + ce_loss_weight * label_loss

# loss.backward()
# optimizer.step()

# running_loss += loss.item()


class DistillationMSELoss:
    def __init__(
        self, feature_map_weight: float = 0.25, standard_targets_weight: float = 0.75
    ):
        self.standard_targets_loss = (
            torch.nn.functional.binary_cross_entropy_with_logits
        )
        self.mse_loss = nn.MSELoss(reduction="mean")
        self.feature_map_weight = feature_map_weight
        self.standard_targets_weight = standard_targets_weight

    def __call__(self, student_logits, teacher_logits, target):
        hidden_rep_loss = self.mse_loss(student_logits, teacher_logits)
        # print("student_logits------>", student_logits.shape)
        # print("teacher_logits------>", teacher_logits.shape)

        # target_loss = self.standard_targets_loss(
        #     student_logits, target, reduction="sum"
        # )

        target_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            student_logits, target, reduction="sum"
        )
        return target_loss
        # print("hidden_rep_loss=---->", hidden_rep_loss)
        # print("target_loss--------->", target_loss)

        return (
            self.feature_map_weight * hidden_rep_loss
            + self.standard_targets_weight * target_loss
        )


class DistillationCrossEntropyLoss:
    def __init__(
        self, temperature: int = 1, alpha: float = 0.7, ignore_index: int = -100
    ) -> None:
        self.standard_targets_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.soft_targets_loss = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.temperature = temperature
        self.alpha = alpha
        self.ignore_index = ignore_index

    def __call__(self, student_output, teacher_output, target):
        # # Resize teacher logits to match the size of student logits
        # resized_teacher_logits = resize_outputs(
        #     teacher_logits, student_logits.shape[2:]
        # )
        # # Resize student logits to match the target size
        # resized_student_logits = resize_outputs(student_logits, target.shape[-2:])
        # student_logits = torch.special.logit(F.sigmoid(student_output))
        # teacher_logits = torch.special.logit(F.sigmoid(teacher_output))
        student_logits = torch.sigmoid(student_output)
        teacher_logits = torch.sigmoid(teacher_output)
        # print("student_logits------>", student_logits.shape, student_logits)
        # print("student_output------>", student_output.shape, student_output)
        # print("teacher_logits------>", teacher_logits.shape, teacher_logits)
        # print("teacher_output------>", teacher_output.shape, teacher_output)
        # student_logits = student_output
        # teacher_logits = teacher_output
        # print("student_logits------>", student_logits.shape, student_logits)
        # print("teacher_logits------>", teacher_logits.shape, teacher_logits)
        # print("target ------------->", target.shape, target)

        # Standard CrossEntropyLoss
        hard_labels_loss = self.standard_targets_loss(student_logits, target)

        # Distillation loss
        soft_labels_loss = self.soft_targets_loss(
            self.log_softmax(student_logits / self.temperature),
            self.softmax(teacher_logits / self.temperature),
        )
        soft_labels_loss = soft_labels_loss * self.temperature**2

        loss = hard_labels_loss + self.alpha * soft_labels_loss
        return loss


class DistillationKLDivLoss:
    def __init__(
        self, temperature: int = 1, alpha: float = 0.7, ignore_index: int = -100
    ) -> None:
        self.standard_targets_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        # self.soft_targets_loss = nn.KLDivLoss(reduction="batchmean")
        self.soft_targets_loss = nn.KLDivLoss(reduction="sum")
        self.softmax = nn.Softmax(dim=1)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.temperature = temperature
        self.alpha = alpha
        self.ignore_index = ignore_index

    def __call__(self, student_logits, teacher_logits, target):
        # # Resize teacher logits to match the size of student logits
        # resized_teacher_logits = resize_outputs(
        #     teacher_logits, student_logits.shape[2:]
        # )
        # # Resize student logits to match the target size
        # resized_student_logits = resize_outputs(student_logits, target.shape[-2:])

        # Standard CrossEntropyLoss
        # hard_labels_loss = self.standard_targets_loss(resized_student_logits, target)
        hard_labels_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            student_logits, target, reduction="sum"
        )

        # Distillation loss
        soft_labels_loss = self.soft_targets_loss(
            self.log_softmax(student_logits / self.temperature),
            self.softmax(teacher_logits / self.temperature),
        )
        soft_labels_loss = soft_labels_loss * self.temperature**2
        soft_labels_loss = soft_labels_loss / (
            student_logits.shape[1] * student_logits.shape[2]
        )

        loss = hard_labels_loss + self.alpha * soft_labels_loss
        return loss


def resize_outputs(
    output: torch.Tensor,
    output_size: Tuple[int, int],
    mode: str = "bilinear",
) -> torch.Tensor:
    """Resizes the outputs to a given size to match the labels.

    Args:
        output (torch.Tensor): model output tensor to be resized
        output_size (Tuple[int, int]): desired size

    Returns:
        torch.Tensor: resized output tensor
    """
    output_resized = nn.functional.interpolate(
        output,
        size=output_size,  # (height, width)
        mode=mode,
        align_corners=False,
    )
    return output_resized
