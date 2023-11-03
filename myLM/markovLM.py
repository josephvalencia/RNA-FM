import torch
import math,time
import random
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

class ByteNetMarkovLM(torch.nn.Module):
    '''Builds a stack of ByteNet layers and a final multi-output regression layer'''

    def __init__(self,k,vocab_size,embed_dim,n_layers=2,downsample=True):
        super().__init__()
        self.downsample = downsample
        self.embedding = torch.nn.Embedding(vocab_size,embed_dim,padding_idx=6)
        self.layers = torch.nn.ModuleList([ByteNetLayer(embed_dim,dilation_rate=1,downsample=downsample) for i in range(n_layers)])
        self.output = torch.nn.Linear(embed_dim*k,vocab_size)
        self.embed_dim = embed_dim

    '''
    def forward(self,x):
        
        x = self.embedding(x).permute(0,2,1)
        for layer in self.layers:
            x = layer(x)
        pred = self.output(x.reshape(x.shape[0],-1))
        return pred
    
    '''

    def forward(self,x):
        
        # x : (B,L)
        B,L = x.shape
        x = self.embedding(x).permute(0,2,1).unsqueeze(-1) # (B,embed_dim,L,1)
        
        # unfold into sliding windows of size k
        x = F.unfold(x,(k,1)).permute(0,2,1) # x : (B,(L-k+1),embed_dim * k)
        x = x.reshape(-1,self.embed_dim,k) # x : (B*(L-k+1),embed_dim,k) 
        
        # reduced seq len is packed into batch dim so kmers are processed independently 
        for layer in self.layers:
            x = layer(x)
        
        pred = self.output(x.reshape(x.shape[0],-1))
        reshaped = pred.reshape(B,(L-k+1),-1)
        return reshaped

def to_nuc(x,y,vocab,limit=10):

    argmax = torch.argmax(x,dim=-1).cpu()
    preds = []
    B,L = argmax.shape
    for i in range(B):
        seq = [vocab[x] for x in argmax[i,:].tolist()]
        preds.append(''.join(seq))

    gtruths = []
    B,L = y.shape
    for i in range(B):
        seq = [vocab[x] for x in y[i,:].tolist()]
        gtruths.append(''.join(seq))

    for i,(pred,gt) in enumerate(zip(preds,gtruths)):
        print(f'batch {i}, pred = {pred[:limit]}... , gt = {gt[:limit]}...')


# hparams
k = 9
dim = 512
batch_size = 8

vocab = {'A' : 0, 
        'C' : 1, 
        'G' : 2, 
        'T' : 3, 
        '<sos>' :4,
        '<eos>' : 5, 
        '<pad>' : 6}

rev_vocab = {v:k for k,v in vocab.items()}

n_nucs = 4
vocab_size = len(vocab)
N = 50

# build dataset
dataset = []
for i in range(N):
    # seq length 
    L = random.randint(800,1200)
    a = torch.randint(0,n_nucs,(batch_size,L))
    padding = torch.zeros(batch_size,k).long()
    start = torch.ones(batch_size,1).long() * vocab['<sos>']
    end = torch.ones(batch_size,1).long() * vocab['<eos>']
    # input x and y are shifted versions of the same seq
    # x has leading padding of length k, 
    x = torch.cat([padding,a],dim=1)
    y = torch.cat([a,end],dim=1)
     
    '''
    # extract sliding windows of size k+1 in the seq dim 
    unfolded = F.unfold(a,(k+1,1)).long().permute(0,2,1)
    # pack the new seq dim into  
    batched = unfolded.reshape(-1,k+1)
    x = batched[:,:-1]
    y = batched[:,-1]
    
    ''' 
    dataset.append((x,y))

net = ByteNetMarkovLM(k,vocab_size,dim)
optimizer = torch.optim.Adam(net.parameters(),lr=1e-3)
nll_loss = torch.nn.NLLLoss(reduction='mean')

s = time.time()
for epoch in range(100):
    running_loss = 0.0
    for i,batch in enumerate(dataset): 
        x,y = batch
        y_pred = net(x)
        logits = F.log_softmax(y_pred,dim=-1)
        #to_nuc(logits,y,rev_vocab)
        # backprop 
        optimizer.zero_grad() 
        loss = nll_loss(logits.reshape(-1,vocab_size),y.reshape(-1))      
        running_loss += loss.item()
        loss.backward()
        l2_norm = lambda x: torch.sqrt(torch.sum(x**2))

        '''
        # monitor gradients
        for name, param in net.named_parameters():
            print(f'norm(grad({name})) = {l2_norm(param.grad)}')
        ''' 
        
        optimizer.step()
    avg_loss = running_loss / len(dataset)
    print(f'epoch={epoch}, i={i}, avg_loss={avg_loss}, ppl={math.exp(avg_loss)}')
        


e = time.time()
print(f'total time: {e-s:.3f}')