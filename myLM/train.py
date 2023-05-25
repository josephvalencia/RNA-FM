import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from myLM.model import RNAAutoregressiveLM
from torch.utils.data import DataLoader
import biotite.sequence.io.fasta as fasta
import numpy as np
import fm
import loralib as lora
from torchdata.datapipes.iter import IterableWrapper
from collections import Counter
from lightning.pytorch.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
from tqdm import tqdm
import torch.nn.functional as F
from functools import partial

def dna_to_rna(entry):
    name,seq = entry 
    return (name,seq.replace('T','U'))

def is_legal_input(entry,alphabet):
    '''ensure all chars are in the vocab and the seq is <=1024 (<cls> + seq + <eos>)''' 
    name,seq = entry 
    return not any([tok not in alphabet.tok_to_idx for tok in seq]) and len(seq) <= 1022

def dataloader_from_fasta(filename,max_tokens,alphabet):
    '''Use datapipes to load sequences from a fasta file into a dataloader.'''
    
    fasta_file = fasta.FastaFile.read(filename) 
    dataset = IterableWrapper(list(fasta_file.items()))
    dataset = dataset.map(dna_to_rna)
    filter_fn = partial(is_legal_input,alphabet=alphabet)
    dataset = dataset.filter(filter_fn)
    dataset = dataset.shuffle(buffer_size=15000)
    dataset = dataset.max_token_bucketize(max_token_count=max_tokens,
                                          max_len=1024,
                                          len_fn=lambda x : len(x[1]),buffer_size=15000)
    dataloader = DataLoader(dataset,batch_size=None,shuffle=True)
    return dataloader

def summarize_dataloader(loader):

    batch_sizes = []
    tokens = []
    for batch in loader:
        batch_sizes.append(len(batch))
        tokens.append(sum([len(seq) for name,seq in batch])) 

    counts = Counter(batch_sizes)
    print(f'Batch size counts: {counts}') 

    mean = np.mean(tokens)
    std = np.std(tokens)
    print(f'Tokens mean: {mean}, std: {std}') 
    print(f'Number of batches: {len(batch_sizes)}')

def make_readable(tokens,idx_to_tok):
    return ["".join([idx_to_tok[x] for x in s]) for s in tokens.cpu().tolist()]

def count_diff(true,pred):
    diff = 0
    #assert len(true) == len(pred)
    for t,p in zip(true,pred):
        if t != p:
            diff += 1
    return diff / len(true)

def reset_model_weights(layer):
    if hasattr(layer, 'reset_parameters'):
        layer.reset_parameters()
    else:
        if hasattr(layer, 'children'):
            for child in layer.children():
                reset_model_weights(child)

def inference(model,alphabet,dataset):

    model.cuda()
    batch_converter = alphabet.get_batch_converter()
    tok_to_idx = alphabet.tok_to_idx
    idx_to_tok = {v:k for k,v in tok_to_idx.items()}
    for batch in tqdm(dataset):
        # prepare batch 
        batch_labels, batch_strs, batch_tokens = batch_converter(batch)
        batch_tokens = batch_tokens.cuda()
        # shifted target for autoregressive LM prediction 
        pad_suffix = torch.ones(batch_tokens.shape[0],1,dtype=torch.long).cuda()*alphabet.padding_idx 
        tgt = torch.cat([batch_tokens[:,1:],pad_suffix],dim=-1)
        
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[12])

        # print results 
        token_logits = results["logits"]
        logprobs = F.log_softmax(token_logits,dim=-1)
        chars = torch.argmax(logprobs,dim=-1)
        readable_pred = make_readable(chars,idx_to_tok)
        readable_true = make_readable(tgt,idx_to_tok)
        for j,(true,pred) in enumerate(zip(readable_true,readable_pred)):
            diff = count_diff(true,pred)
            print(f' {j}, diff = {diff}, true = {true}, pred = {pred}')

if __name__ == "__main__":

    train_file = "/nfs6/BB/Hendrix_Lab/valejose/bioseq2seq/data/mammalian_200-1200_train_RNA_balanced.fa"
    test_file = "/nfs6/BB/Hendrix_Lab/valejose/bioseq2seq/data/mammalian_200-1200_test_RNA_nonredundant_80.fa"
    val_file = "/nfs6/BB/Hendrix_Lab/valejose/bioseq2seq/data/mammalian_200-1200_val_RNA_nonredundant_80.fa"

    stop_metric = "val_loss"
    checkpoint_callback = ModelCheckpoint(
                        dirpath="myLM/checkpoints",
                        monitor=stop_metric,
                        save_top_k=3,
                        every_n_epochs=1,
                        filename="{epoch}-{val_loss:.4f}-{val_acc:.3f}")

    early_stopping = EarlyStopping(stop_metric,patience=2)

    # Load RNA-FM model
    override_args = {"q_lora_rank" : 64, "v_lora_rank" : 64}
    model, alphabet = fm.pretrained.rna_fm_t12(override_args=override_args)
    lora.mark_only_lora_as_trainable(model)
    #reset_model_weights(model)
    
    # load datasets 
    max_tokens = 1024*16
    train_loader = dataloader_from_fasta(train_file,max_tokens,alphabet)
    val_loader = dataloader_from_fasta(val_file,max_tokens,alphabet) 
    test_loader = dataloader_from_fasta(test_file,max_tokens,alphabet)
    summarize_dataloader(train_loader)

    # train the model
    module = RNAAutoregressiveLM(model,alphabet) 
    wandb_logger = pl.loggers.WandbLogger(project="rna-lm")
    trainer = pl.Trainer(max_epochs=1,
                         devices=4,
                         accelerator="gpu",
                         strategy="deepspeed_stage_3_offload",
                         logger=wandb_logger, 
                         precision='16-mixed',
                         callbacks=[checkpoint_callback])
    trainer.fit(module,train_loader,val_loader)
    
    # load best model and run inference 
    best_model_chkpt = "myLM/checkpoints/best_model.pt"
    convert_zero_checkpoint_to_fp32_state_dict(checkpoint_callback.best_model_path, best_model_chkpt)
    trained_state_dict = torch.load(best_model_chkpt,map_location=torch.device('cpu'))['state_dict']
    trained_state_dict = {k.replace("model.","") : v for k,v in trained_state_dict.items()}
    model, alphabet = fm.pretrained.rna_fm_t12(override_args=override_args)
    model.load_state_dict(trained_state_dict,strict=True) 
    inference(model,alphabet,train_loader)
