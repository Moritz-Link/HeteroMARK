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

    log_prob = dist.log_prob(logits[0, 0])
    print("Log Prob:", logits[0, 0], log_prob)

    print("\n", " === Gradient Check === ")
    logits.requires_grad_(True)

    dist = RsampleCategorical(logits=logits)
    sample = dist.rsample(hard=True, tau=0.5)
    loss = sample.sum()
    loss.backward()
    print("Logits Grad:", logits.grad)
    print("\n", " === Masked Rsample Categorical Test === ")


if __name__ == "__main__":
    test_rsample_categorical()
