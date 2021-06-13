import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from model.resnet_model import init_model
from model.keypoint_model import KP_Orientation_Net

class Model(nn.Module):
  def __init__(self, name, num_classes, loss, pretrained, use_gpu):
    super(Model, self).__init__()
    self.resnet50 = init_model(name=name, num_classes=num_classes, loss=loss,
                              pretrained=pretrained, use_gpu=use_gpu)
    self.kp_net = KP_Orientation_Net.KeyPointModel()
    self.img_rnn1 = nn.LSTM(256, 256, 1, batch_first=True)
    self.img_rnn2 = nn.LSTM(768, 768, 1, batch_first=True)
    self.img_rnn3 = nn.LSTM(1792, 1792, 1, batch_first=True)
    self.AttentionLayer = AttentionLayer()
    self.dropout = nn.Dropout(0.5)
    self.final_mlp = nn.Linear(1792, 512)

  def forward(self, x, patchs):
      y, v, x1, x2, x3 = self.resnet50(x)
      N1, C1, H1, W1 = x1.shape
      N2, C2, H2, W2 = x2.shape
      N3, C3, H3, W3 = x3.shape
      patchs_N, p, four = patchs.shape
      local1 = torch.autograd.Variable(torch.zeros((N1, p, C1, 1, 1)).cuda())
      local2 = torch.autograd.Variable(torch.zeros((N2, p, C2, 1, 1)).cuda())
      local3 = torch.autograd.Variable(torch.zeros((N3, p, C3, 1, 1)).cuda())
      for i in range(N1):
          seven = patchs[i]
          for k in range(p):
              if seven[k][0] == 0 and seven[k][1] == 0 and seven[k][2] == 0 and seven[k][3] == 0:
                  local1[i][k] = 0
              else:
                  feat = x1[i, :, int(seven[k][0] * 55 / 55):int(seven[k][1] * 55 / 55) + 1,
                         int(seven[k][2] * 55 / 55):int(seven[k][3] * 55 / 55) + 1]
                  local1[i][k] = F.avg_pool2d(feat, feat.size()[1:])
      local1 = local1.view(local1.size(0), local1.size(1), -1)
      rnn_img1, (hidden_state1, c_state1) = self.img_rnn1(local1)
      for i in range(N2):
          seven = patchs[i]
          for k in range(p):
              if seven[k][0] == 0 and seven[k][1] == 0 and seven[k][2] == 0 and seven[k][3] == 0:
                  local2[i][k] = 0
              else:
                  feat = x2[i, :, int(seven[k][0] * 27 / 55):int(seven[k][1] * 27 / 55) + 1,
                         int(seven[k][2] * 27 / 55):int(seven[k][3] * 27 / 55) + 1]  # 2048*16*16
                  local2[i][k] = F.avg_pool2d(feat, feat.size()[1:])
      local2 = local2.view(local2.size(0), local2.size(1), -1)
      local2 = torch.cat((local2, rnn_img1), 2)
      rnn_img2, (hidden_state2, c_state2) = self.img_rnn2(local2)
      for i in range(N3):
          seven = patchs[i]
          for k in range(p):
              if seven[k][0] == 0 and seven[k][1] == 0 and seven[k][2] == 0 and seven[k][3] == 0:
                  local3[i][k] = 0
              else:
                  feat = x3[i, :, int(seven[k][0] * 13 / 55):int(seven[k][1] * 13 / 55) + 1,
                         int(seven[k][2] * 13 / 55):int(seven[k][3] * 13 / 55) + 1]  # 2048*16*16
                  local3[i][k] = F.avg_pool2d(feat, feat.size()[1:])
      local3 = local3.view(local3.size(0), local3.size(1), -1)
      local3 = torch.cat((local3, rnn_img2), 2)
      rnn_img3, (hidden_state3, c_state3) = self.img_rnn3(local3)
      attn_lstm_emb = self.AttentionLayer.forward(rnn_img3)
      batch_size = attn_lstm_emb.shape[0]
      attn_lstm_emb = attn_lstm_emb.view(batch_size, -1).contiguous()
      attn_lstm_emb = self.dropout(attn_lstm_emb)
      GCN_feature = self.final_mlp(attn_lstm_emb)
      if not self.training:
          addfeature = torch.cat((v, GCN_feature), 1)
          return addfeature
      return y, v, GCN_feature


class AttentionLayer(nn.Module):

    def __init__(self, seq_len=7, hidden_emb=1792):
        super(AttentionLayer, self).__init__()

        self.seq_len = seq_len
        self.hidden_emb = hidden_emb
        self.mlp1_units = 3584
        self.mlp2_units = 1792

        self.fc = nn.Sequential(
            nn.Linear(self.seq_len*self.hidden_emb, self.mlp1_units),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(self.mlp1_units, self.mlp2_units),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(self.mlp2_units, self.seq_len),
            nn.ReLU(inplace=True),
            )

        self.softmax = nn.Softmax(dim=1)

    '''
    Computes the attention on the lstm outputs from each time step in the sequence
    Arguments:
        lstm_emd : lstm embedding from each time step in the sequence
    Returns:
        attn_feature_map : embedding computed after applying attention to the lstm embedding of the entire sequence
    '''
    def forward(self, lstm_emd): 

        batch_size = lstm_emd.shape[0]
        lstm_emd = lstm_emd.contiguous()
        lstm_flattened = lstm_emd.view(batch_size, -1) # to pass it to the MLP architecture

        attn = self.fc(lstm_flattened) # attention over the sequence length
        alpha = self.softmax(attn) # gives the probability values for the time steps in the sequence (weights to each time step)
        #print(alpha.shape)
        alpha = torch.stack([alpha]*self.mlp2_units, dim=2) # stack it across the lstm embedding dimesion

        attn_feature_map = lstm_emd * alpha # gives attention weighted lstm embedding
        attn_feature_map = torch.sum(attn_feature_map, dim=1, keepdim=True) # computes the weighted sum
        return attn_feature_map