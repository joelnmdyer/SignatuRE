from torch import nn

class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, **kwargs):
        super(GRU, self).__init__(**kwargs)

        self.mod = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.mod(x)
        out = out[:, -1, :]
        return self.fc(out)
