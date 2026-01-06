# Categorical
import torch
import torch.nn.functional as F
from torchrl.modules import (
    MaskedCategorical,
)


class RsampleCategorical(torch.distributions.Categorical):
    def rsample(self, hard=True, tau=1.0):
        return F.gumbel_softmax(self.logits, hard=hard, tau=tau)


class MaskedRsampleCategorical(MaskedCategorical):
    def __init__(
        self,
        logits: torch.Tensor | None = None,
        probs: torch.Tensor | None = None,
        *,
        mask: torch.Tensor | None = None,
        indices: torch.Tensor | None = None,
        neg_inf: float = float("-inf"),
        padding_value: int | None = None,
        use_cross_entropy: bool = True,
        padding_side: str = "left",
    ) -> None:
        super().__init__(
            logits=logits,
            probs=probs,
            mask=mask,
            indices=indices,
            neg_inf=neg_inf,
            padding_value=padding_value,
            use_cross_entropy=use_cross_entropy,
            padding_side=padding_side,
        )

    def rsample(self, hard=True, tau=1.0):
        return F.gumbel_softmax(self.logits, hard=hard, tau=tau)


def test_rsample_categorical():
    print(" === Rsample Categorical Test === ")
    logits = torch.tensor([[0.1, 0.2, 0.7], [0.3, 0.4, 0.3]])
    dist = RsampleCategorical(logits=logits)
    print("Logits:", dist.logits, dist.logits.shape)
    sample = dist.rsample(hard=True, tau=0.5)
    print("Sample:", sample, sample.shape)

    actions = sample.argmax(dim=-1)  # shape [2]
    print("Actions:", actions, actions.shape)
    log_prob = dist.log_prob(actions)
    print("Log Prob:", actions, log_prob)

    print("\n", " = Gradient Check =")
    logits.requires_grad_(True)

    dist = RsampleCategorical(logits=logits)
    sample = dist.rsample(hard=True, tau=0.5)
    loss = sample.sum()
    loss.backward()
    print("Logits Grad:", logits.grad)


def test_masked_rsample_categorical():
    print("\n", " === Masked Rsample Categorical Test === ")
    logits = torch.tensor([[0.1, 0.2, 0.7], [0.3, 0.4, 0.3]])
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)
    dist = MaskedRsampleCategorical(logits=logits, mask=mask)
    print("Logits:", dist.logits, dist.logits.shape)
    sample = dist.rsample(hard=True, tau=0.5)
    print("Sample:", sample, sample.shape)

    actions = sample.argmax(dim=-1)  # shape [2]
    print("Actions:", actions, actions.shape)
    log_prob = dist.log_prob(actions)
    print("Log Prob:", actions, log_prob)

    print("\n", " = Gradient Check =")
    logits.requires_grad_(True)

    dist = MaskedRsampleCategorical(logits=logits, mask=mask)
    sample = dist.rsample(hard=True, tau=0.5)
    loss = sample.sum()
    loss.backward()
    print("Logits Grad:", logits.grad)


if __name__ == "__main__":
    # test_rsample_categorical()
    test_masked_rsample_categorical()
