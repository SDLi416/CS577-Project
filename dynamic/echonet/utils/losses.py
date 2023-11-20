import torch
import torch.nn as nn

# refer to https://github.com/JakubTomaszewski/Knowledge-Distillation-in-Semantic-Segmentation/blob/main/src/knowledge_distillation/losses.py#L37


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
