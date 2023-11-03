import torch
import torch.nn as nn
import torch.nn.functional as F

class ByteNetLayer(torch.nn.Module):
    '''PyTorch implementation of ByteNet from https://arxiv.org/abs/1610.10099
      as modified in https://openreview.net/pdf?id=3i7WPak2sCx'''
    
    def __init__(self,dim=128,dilation_rate=1,downsample=False,dropout=0.4):
        super().__init__()

        lower_dim = dim // 2 if downsample else dim
        self.layernorm1 = nn.InstanceNorm1d(dim,affine=True)
        self.layernorm2 = nn.InstanceNorm1d(lower_dim,affine=True)
        self.layernorm3 = nn.InstanceNorm1d(lower_dim,affine=True)
        self.dropout = nn.Dropout(dropout)

        self.cnn1 = nn.Conv1d(in_channels=dim,
                              out_channels=lower_dim,
                              kernel_size=1)

        self.cnn2 = nn.Conv1d(in_channels=lower_dim,
                              out_channels=lower_dim,
                              kernel_size=5,
                              dilation=dilation_rate,
                              padding='same')
        
        self.cnn3 = nn.Conv1d(in_channels=lower_dim,
                              out_channels=dim,
                              kernel_size=1)

    def forward(self,x):
        '''x : torch.Tensor of shape (batch_size,embedding_size,sequence_length)'''
        
        residual = x  
        x = self.layernorm1(x)
        x = F.gelu(x)
        x = self.cnn1(x)
        x = self.layernorm2(x)
        x = F.gelu(x)
        x = self.cnn2(x)
        x = self.layernorm3(x)
        x = F.gelu(x)
        x = self.cnn3(x)
        return self.dropout(x)+residual

class ByteNetRegression(torch.nn.Module):
    '''Builds a stack of ByteNet layers and a final multi-output regression layer'''

    def __init__(self,n_outputs,embed_dim,n_layers=10,downsample=True):
        super().__init__()
        self.downsample = downsample
        #self.embedding = nn.Linear(6,embed_dim)
        self.embedding = nn.Embedding(6,embed_dim,padding_idx=5)
        self.layers = nn.ModuleList([ByteNetLayer(embed_dim,dilation_rate=2**(i%5),downsample=downsample) for i in range(n_layers)])
        self.regression = nn.Linear(embed_dim,n_outputs)
        #self.out_lstm = nn.LSTM(embed_dim,embed_dim //2,batch_first=True,bidirectional=True)

    def forward(self,x):
        '''x : torch.Tensor of shape (batch_size,embedding_size,sequence_length)'''
        x = self.embedding(x).permute(0,2,1)
    
        for layer in self.layers:
            x = layer(x)
        #seq_embed = self.out_lstm(x.permute(0,2,1))[0]
        #x = self.regression(seq_embed[:,0,:]).squeeze(1)
        seq_embed = x.permute(0,2,1)
        x = self.regression(seq_embed.mean(dim=1)).squeeze(1)
        return x