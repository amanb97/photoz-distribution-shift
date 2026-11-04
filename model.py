import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical, Normal, MixtureSameFamily

# defining MDN
class MDN(nn.Module):
    def __init__(
        self,
        input_dim=5,
        hidden_dim=64,
        n_components=5,
        min_sigma=1e-3,
    ):
        super().__init__()

        self.n_components = n_components
        self.min_sigma = min_sigma

        #define the backbone
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )


        #produces the parameters including means, scales and weights
        self.mean_head = nn.Linear(hidden_dim, n_components)
        self.scale_head = nn.Linear(hidden_dim, n_components)
        self.weight_head = nn.Linear(hidden_dim, n_components)

    def forward(self, x):
        #pass the input through
        hidden = self.backbone(x)

        means = self.mean_head(hidden)
        scales = F.softplus(self.scale_head(hidden)) + self.min_sigma
        logits = self.weight_head(hidden)

        return means, scales, logits

    def distribution(self, x):
        #builds the distribution from params
        means, scales, logits = self(x)

        components = Normal(loc=means, scale=scales)
        mixture_weights = Categorical(logits=logits)

        return MixtureSameFamily(mixture_weights, components)


def mdn_loss(distribution, targets):
    """Mean negative log-likelihood of the reference redshifts."""
    return -distribution.log_prob(targets).mean()


if __name__ == "__main__":
    torch.manual_seed(42)

    model = MDN()

    # Artificial inputs to check the model's shapes and calculations.
    features = torch.randn(8, 5)
    targets = torch.linspace(0.1, 0.6, steps=8)

    means, scales, logits = model(features)
    distribution = model.distribution(features)

    loss = mdn_loss(distribution, targets)
    loss.backward()

    print("Means:", means.shape)
    print("Scales:", scales.shape)
    print("Weights:", logits.softmax(dim=-1).shape)

    print("Weights sum:", logits.softmax(dim=-1).sum(dim=-1))
    print("Smallest scale:", scales.min().item())
    print("Loss:", loss.item())
    print("Predicted mean:", distribution.mean.shape)

    assert torch.isfinite(loss), "Loss is not finite."
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters()
    ), "Missing or non-finite gradients."