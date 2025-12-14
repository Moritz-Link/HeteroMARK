import torch


class SmallMlpCritic(torch.nn.Module):
    def __init__(self, input_dim: int):
        super(SmallMlpCritic, self).__init__()
        self.input_dim = input_dim
        layers = []
        last_size = input_dim

        for size in [64, 64]:
            layers.append(torch.nn.Linear(last_size, size))
            layers.append(torch.nn.ReLU())
            last_size = size
        layers.append(torch.nn.Linear(last_size, 1))  # Output layer
        self.model = torch.nn.Sequential(*layers)

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        # X = [arg.flatten(start_dim=1) for arg in args]
        if args[0].dim() == 2:
            X = [arg.flatten(0) for arg in args]
        elif args[0].dim() == 3:
            X = [arg.flatten(1) for arg in args]
        elif args[0].dim() == 4:
            X = [arg.flatten(2) for arg in args]
        x = torch.cat(X, dim=-1)
        return self.model(x)


if __name__ == "__main__":
    # Example usage
    modl = SmallMlpCritic(input_dim=25)
    a = torch.zeros(2, 128, 3, 5)
    b = torch.zeros(2, 128, 2, 5)
    a = modl(a, b)
    print(a.shape)  # Should output torch.Size([128, 1])
# print(modl(a, b))  # Should output a tensor of shape (1, 1)
