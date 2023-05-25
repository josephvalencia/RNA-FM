from typing import Any, Dict
import torch
from torch import optim
import lightning.pytorch as pl
import torch.nn.functional as F
import fm
import loralib as lora
from deepspeed.ops.adam import DeepSpeedCPUAdam
import torchmetrics

# define the LightningModule
class RNAAutoregressiveLM(pl.LightningModule):
    
    def __init__(self, model : fm.RNABertModel , alphabet : fm.Alphabet):
        super().__init__()
        self.model = model
        self.alphabet = alphabet
        vocab_size = len(alphabet.all_toks)
        print(f'Vocab size: {vocab_size}')
        self.batch_converter = alphabet.get_batch_converter()
        self.nll_loss = torch.nn.NLLLoss(ignore_index=alphabet.padding_idx, reduction='mean')
        self.train_perplexity = torchmetrics.Perplexity(ignore_index=alphabet.padding_idx)
        self.val_perplexity = torchmetrics.Perplexity(ignore_index=alphabet.padding_idx)
        self.train_accuracy = torchmetrics.Accuracy(task='multiclass',num_classes=vocab_size)
        self.val_accuracy = torchmetrics.Accuracy(task='multiclass',num_classes=vocab_size)

    def forward(self,x):
        return self.model(x, repr_layers=[12],checkpoint_activations=True)
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint["state_dict"] = lora.lora_state_dict(self.model)

    def lm_target(self,src):
        # shifted target for autoregressive LM prediction 
        pad_suffix = torch.ones(src.shape[0],1,dtype=torch.long).cuda()*self.alphabet.padding_idx 
        return torch.cat([src[:,1:],pad_suffix],dim=-1) 

    def shared_step(self,batch,batch_idx):
        batch_labels, batch_strs, batch_tokens = self.batch_converter(batch)
        batch_tokens = batch_tokens.to(self.device) 
        results = self.model(batch_tokens, repr_layers=[12],checkpoint_activations=True)
        token_logits = results["logits"]
        # compute loss and backprop 
        logprobs = F.log_softmax(token_logits,dim=-1)
        tgt = self.lm_target(batch_tokens) 
        lm_loss = self.nll_loss(logprobs.view(-1,logprobs.shape[-1]),tgt.view(-1)) 
        return lm_loss,logprobs,tgt

    def training_step(self, batch, batch_idx):
        lm_loss,logprobs,tgt = self.shared_step(batch,batch_idx)
        pred = torch.argmax(logprobs,dim=-1) 
        self.log("train_loss", lm_loss,batch_size=len(batch))
        self.log("train_ppl", self.train_perplexity(logprobs.float(),tgt),batch_size=len(batch))
        self.log("train_acc", self.train_accuracy(pred,tgt),batch_size=len(batch))
        return lm_loss
    
    def validation_step(self, batch, batch_idx):
        lm_loss,logprobs,tgt = self.shared_step(batch,batch_idx) 
        pred = torch.argmax(logprobs,dim=-1) 
        self.log("val_loss", lm_loss,batch_size=len(batch),sync_dist=True)
        self.log("val_ppl", self.val_perplexity(logprobs.float(),tgt),batch_size=len(batch),sync_dist=True)
        self.log("val_acc", self.val_accuracy(pred,tgt),batch_size=len(batch),sync_dist=True)

    def configure_optimizers(self):
        #optimizer = optim.Adam(self.parameters(), lr=1e-3)
        optimizer = DeepSpeedCPUAdam(self.parameters(), lr=1e-3) 
        return optimizer


