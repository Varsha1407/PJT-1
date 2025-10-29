import torch
import torch.nn as nn
import torch.nn.functional as F

class MultimodalTemporalFusion(nn.Module):
    def __init__(self, num_classes=5, dropout_rate=0.5):
        super(MultimodalTemporalFusion, self).__init__()
        
        # ===== SPATIAL ENCODERS (3D CNN for each modality) =====
        
        # RGB Stream: 3D CNN (3 channels)
        self.rgb_conv1 = nn.Conv3d(3, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.rgb_conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        
        # Depth Stream: 3D CNN (1 channel)
        self.depth_conv1 = nn.Conv3d(1, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.depth_conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        
        # Thermal Stream: 3D CNN (1 channel)
        self.thermal_conv1 = nn.Conv3d(1, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.thermal_conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        
        self.pool3d = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.dropout = nn.Dropout3d(dropout_rate)
        
        # ===== GLOBAL AVERAGE POOLING =====
        # Output: (batch, 64, 1, 1, 1) per stream
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # ===== TEMPORAL FUSION WITH LSTM =====
        # Feature dimensions: 64 per modality
        # Concatenated: 64*3 = 192 features per frame
        self.lstm = nn.LSTM(input_size=192, hidden_size=128, num_layers=2, 
                            batch_first=True, dropout=dropout_rate)
        
        # ===== TEMPORAL ATTENTION =====
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        
        # ===== CLASSIFICATION HEAD =====
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, rgb, depth, thermal):
        """
        Input shapes:
        rgb: (batch, 3, T, H, W)
        depth: (batch, 1, T, H, W)
        thermal: (batch, 1, T, H, W)
        """
        
        # ===== RGB STREAM =====
        rgb_x = self.relu(self.rgb_conv1(rgb))
        rgb_x = self.pool3d(rgb_x)
        rgb_x = self.dropout(rgb_x)
        rgb_x = self.relu(self.rgb_conv2(rgb_x))
        rgb_x = self.pool3d(rgb_x)
        rgb_x = self.global_pool(rgb_x).squeeze(-1).squeeze(-1).squeeze(-1)  # (batch, 64)
        
        # ===== DEPTH STREAM =====
        depth_x = self.relu(self.depth_conv1(depth))
        depth_x = self.pool3d(depth_x)
        depth_x = self.dropout(depth_x)
        depth_x = self.relu(self.depth_conv2(depth_x))
        depth_x = self.pool3d(depth_x)
        depth_x = self.global_pool(depth_x).squeeze(-1).squeeze(-1).squeeze(-1)  # (batch, 64)
        
        # ===== THERMAL STREAM =====
        thermal_x = self.relu(self.thermal_conv1(thermal))
        thermal_x = self.pool3d(thermal_x)
        thermal_x = self.dropout(thermal_x)
        thermal_x = self.relu(self.thermal_conv2(thermal_x))
        thermal_x = self.pool3d(thermal_x)
        thermal_x = self.global_pool(thermal_x).squeeze(-1).squeeze(-1).squeeze(-1)  # (batch, 64)
        
        # ===== EARLY FUSION =====
        # Concatenate features from all modalities
        fused = torch.cat([rgb_x, depth_x, thermal_x], dim=1)  # (batch, 192)
        
        # ===== TEMPORAL LSTM =====
        lstm_out, (h_n, c_n) = self.lstm(fused.unsqueeze(1))  # (batch, 1, 128)
        
        # ===== TEMPORAL ATTENTION =====
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # ===== CLASSIFICATION =====
        output = self.relu(self.fc1(attn_out.squeeze(1)))
        output = self.fc2(output)
        output = self.softmax(output)
        
        return output

# Initialize model
device = torch.device('cpu')  # CPU for your setup
model = MultimodalTemporalFusion(num_classes=5).to(device)
