import torch
import torch.nn as nn
import torch.nn.functional as F


class CorrectedEEG2ClassificationCNNLSTM(nn.Module):
    def __init__(self, input_size=1125, output_size=4, hidden_size=256):
        super(CorrectedEEG2ClassificationCNNLSTM, self).__init__()
        # Spatial Convolution Layer
        self.spatial_conv = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=(1, 3), stride=1, padding=(0, 1)
        )
        # Temporal Convolution Layer
        self.temp_conv = nn.Conv1d(
            in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2
        )

        self.lstm = nn.LSTM(
            input_size=32,  # Correctly adjusted for the flattened output of spatial dimensions
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(hidden_size, output_size)

        # Weight Initialization
        self.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.LSTM or type(m) == nn.Conv1d or type(m) == nn.Conv2d:
            for name, param in m.named_parameters():
                if "weight" in name:
                    nn.init.xavier_uniform_(param.data)
                elif "bias" in name:
                    param.data.fill_(0)
        elif type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0)

    def forward(self, x):
        # Apply spatial convolution
        # import pdb

        # pdb.set_trace()
        x = self.spatial_conv(x)
        x = F.relu(x)

        # Collapse the spatial dimensions and prepare for temporal convolution
        x = x.view(
            x.size(0), x.size(1), -1
        )  # Flatten spatial dimensions except batch size and time
        x = F.relu(self.temp_conv(x))

        # Prepare for LSTM
        x = x.transpose(1, 2)  # Swap the channel and sequence dimensions
        x = x.reshape(x.size(0), x.size(1), -1)  # Correct reshaping for LSTM input

        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        x = self.fc1(self.dropout1(lstm_out))
        return x


class LightWeightEEG2ClassificationCNN(nn.Module):
    def __init__(self, input_size=1125, output_size=4):
        super(LightWeightEEG2ClassificationCNN, self).__init__()
        # Spatial Convolution Layer
        self.spatial_conv = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=(1, 3), stride=1, padding=(0, 1)
        )
        # Temporal Convolution Layer
        self.temp_conv = nn.Conv1d(
            in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2
        )

        # Flatten the output from the temporal convolution to connect to a dense layer
        self.flatten = nn.Flatten()

        # Fully connected layers
        self.fc1 = nn.Linear(792000, 256)  # Adjust the input features accordingly
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, output_size)

        # Weight Initialization
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0)

    def forward(self, x):
        x = self.spatial_conv(x)
        x = F.relu(x)
        x = x.view(x.size(0), x.size(1), -1)  # Flatten spatial dimensions
        x = F.relu(self.temp_conv(x))
        x = self.flatten(
            x
        )  # Flatten all features to prepare for the fully connected layer
        x = F.relu(self.fc1(self.dropout1(x)))
        x = self.fc2(x)  # Output is raw logits for classification
        return x
