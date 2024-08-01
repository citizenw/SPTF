import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, dilation=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, 3, padding=dilation, dilation=dilation)
        self.relu = nn.ReLU()
        self.skip = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        if self.skip is not None:
            x = self.skip(x)
        out += x
        out = self.relu(out)
        return out


class TCN(nn.Module):
    def __init__(self, input_size, output_size, n_layers, n_units, dropout):
        super(TCN, self).__init__()
        self.n_layers = n_layers
        self.conv1 = nn.Conv1d(input_size, n_units, kernel_size=1)
        self.res_blocks = nn.ModuleList([ResidualBlock(n_units, n_units, dilation=2 ** i) for i in range(n_layers)])
        self.conv2 = nn.Conv1d(n_units, output_size, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.skip = nn.Conv1d(input_size, output_size, 1) if input_size != output_size else None

    def forward(self, x):
        out = self.relu(self.conv1(x))
        for i in range(self.n_layers):
            out = self.res_blocks[i](out)
        out = self.dropout(out)
        out = self.conv2(out)
        if self.skip is not None:
            x = self.skip(x)

        return out + x



