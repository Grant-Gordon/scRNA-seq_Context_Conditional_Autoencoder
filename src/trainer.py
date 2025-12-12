#trainer.py 
import torch
import torch.optim as optim
from torch import Tensor
import time
from torch.utils.data import DataLoader
import torch.nn as nn
from models.CAE import CAE
from models.context_classifier import ContextClassifier
from dataloader import SingleChunkDataset, ChunksDataset
from typing import Dict, Tuple, List, Any
import random
from collections import defaultdict
import json
import logging_helpers as log

class Trainer():
    def __init__(self,
                output_dir,
                data_dir="/mnt/projects/debruinz_project/july2024_census_data/subset",
                expr_glob="human_counts_?.npz", #NOTE: Glob uses ? not * for hyphenated training
                meta_glob="human_metadata_?.pkl", #NOTE: Glob uses ? not * for hyphenated training
                field_specs_path="./metadata_vocab.json" ,
                meta_fields_vocabs_path="./metadata_field_specs.json",
                learning_rate=0.001,
                batch_size = 128,
                latent_dim = 128,
                classifier_latent_dim = 128,
                batch_workers = 1,
                batch_prefetch_factor=1
                ):
        t0_init = time.time()
        self.output_dir=output_dir
        self.data_dir= data_dir
        self.expr_glob = expr_glob
        self.meta_glob = meta_glob
        self.field_specs_path = field_specs_path
        self.metadata_fields_vocabs_path = meta_fields_vocabs_path
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.latent_dim = latent_dim  
        self.classifier_latent_dim = classifier_latent_dim  
        self.batch_workers = batch_workers
        self.batch_prefetch_factor = batch_prefetch_factor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tbwriter = log.init_logging(self.output_dir)
        self.log_head_influence=False
        self.head_logit_l2_ems = {}
        
        #Load in JSONs
        with open (self.field_specs_path) as f:
            self.field_specs_dict = json.load(f)
        with open(meta_fields_vocabs_path) as f:
            self.metadata_fields_vocabs = json.load(f)
        print(f"Successfully loaded metadata JSON files - Inside Trainer.__init__()")



        #Dataloader for Chunks 
        chunks_dataset = ChunksDataset(self.data_dir, meta_glob_pattern=self.meta_glob, gene_expr_glob_pattern=self.expr_glob)
        assert len(chunks_dataset) > 0, "No chunks found, Dataset empty"
        first_csr, _ = chunks_dataset[0]
        input_dim = int(first_csr.shape[1])


        #Model and Optimizer
        self.model = CAE(input_dim, self.latent_dim, self.field_specs_dict).to(self.device, dtype=torch.float32)
        #######Create Param groups for optimizer
        base_weight_decay = 1e-4
        meta_weight_decay = 5e-4
        base_params = []
        meta_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "shared_meta_encoders" in name or "shared_meta_decoders" in name or "field_context_head_pool" in name:
                meta_params.append(param)
            else:
                base_params.append(param)

        param_groups =[]
        if base_params:
            param_groups.append({"params": base_params, "weight_decay": base_weight_decay})
        if meta_params:
            param_groups.append({"params": meta_params, "weight_decay":meta_weight_decay})

        self.generator_optimizer = optim.AdamW(param_groups, lr = self.learning_rate)


        
        self.classifier = ContextClassifier(input_dim, self.classifier_latent_dim, self.field_specs_dict).to(self.device)
        self.classifier_optimizer = optim.Adam(self.classifier.parameters(), lr = self.learning_rate)

          
        self.outer_loader = DataLoader(
            chunks_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=1,
            prefetch_factor=1,
            persistent_workers=True,
            pin_memory=(torch.cuda.is_available()),
            collate_fn=lambda batch: batch[0],  #unwraps List: [(csr, meta)] into Tuple: (csr, meta)
            )
        print("Succesfully created model, optim, and outer_laoder - Inside Trainer.__init__()")
        t1_init = time.time() - t0_init
        print(f"[Time Initialing]: {t1_init}, [Current Time]: {time.time()}")
    
    def train(self, num_epochs:int): #TODO: SOmething not right with GPU, check dcgm. 
        t0_train = time.time() #TODO add timing summaries in logging_helpers
        self.model.train()

        #Establish Global Logging
        self.chunks_trained_on = 0 

        for epoch in range(num_epochs):
            print(f"\nBeggining epoch: {epoch} - Time: ~{time.time() - t0_train}")
            #Establish Epoch Logging
            self.sum_chunk_train_times=0.0
            self.epoch_raw_loss_terms = defaultdict(float)
            self.epoch_normed_loss_terms = defaultdict(float)
            self.epoch_raw_adv_field_loss = defaultdict(float)
            self.epoch_classifier_loss = 0.0
            self.chunk_num_in_epoch = 0
            if epoch == num_epochs - 1: 
                self.log_head_influence=True
                self.head_logit_l2_ems = {field: log.Ems() for field in self.model.used_fields}
                self.head_logit_l2_ems["base"] = log.Ems()
            t0_epoch = time.time()  ###RESET LOGS
            
            
            #loop chunks
            for expr_csr_chunk, meta_chunk in self.outer_loader:
                #Establish Chunk Logging 
                self.chunk_num_in_epoch+=1 ###ITERATE LOGS
                self.chunks_trained_on+=1
                self.batch_num_in_chunk=0 ###RESET LOGS
                self.sum_batch_train_times=0.0
                self.chunk_raw_loss_terms = defaultdict(float)
                self.chunk_raw_adv_field_loss = defaultdict(float)
                self.chunk_normed_loss_terms = defaultdict(float)
                self.grad_ems = log.Ems(use_max=True, use_min=True, use_mean=True, use_mode=False, use_median=False)
                print(f"\tTraining {self.chunk_num_in_epoch}th Chunk in Epoch: {epoch}")
                t0_chunk = time.time()

                #Create Datalaoder to prelaod batches from Chunk
                inner_dataset =  SingleChunkDataset((expr_csr_chunk, meta_chunk),field_specs=self.field_specs_dict, used_fields=self.model.used_fields, field_value_map=self.metadata_fields_vocabs)
                inner_loader = DataLoader(
                    dataset=inner_dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=self.batch_workers,
                    prefetch_factor=self.batch_prefetch_factor,
                    pin_memory=(self.device.type == "cuda"),
                    drop_last=False
                )
                #loop batches 
                for expr_batch, meta_batches in inner_loader:
                    t0_batch = time.time()
                    #Move to GPU (supposedly#TODO)
                    expr_batch = expr_batch.to(self.device, dtype=torch.float32, non_blocking=True)
                    meta_batches = {k: v.to(self.device, dtype=torch.float32, non_blocking=True)for k,v in meta_batches.items()}
                    
                    #Train Batch 
                    batch_raw_loss_terms, batch_normed_loss_terms, raw_adv_field_loss = self.train_on_batch(expr_batch, meta_batches)
                    #Compute Logs
                    self.sum_batch_train_times += time.time() - t0_batch
                    self.batch_num_in_chunk+=1
                    #Chunk loss logging
                    for key in batch_raw_loss_terms.keys():
                        raw_v = batch_raw_loss_terms[key]
                        self.chunk_raw_loss_terms[key] += float(raw_v) if torch.is_tensor(raw_v) else float(raw_v)
                        if key not in batch_normed_loss_terms: continue
                        norm_v = batch_normed_loss_terms[key]
                        self.chunk_normed_loss_terms[key] += float(norm_v) if torch.is_tensor(norm_v) else float(norm_v)
                
                    for field, loss in raw_adv_field_loss.items():
                        self.chunk_raw_adv_field_loss[field] +=float(loss) if torch .is_tensor(loss) else float(loss)
                    #IN SCOPE BATCH
                #IN SCOPE CHUNK
                log.per_chunk_raw_loss(self.tbwriter, self.chunks_trained_on, dict(self.chunk_raw_loss_terms))
                log.per_chunk_normed_loss(self.tbwriter, self.chunks_trained_on, dict(self.chunk_normed_loss_terms))
                log.per_chunk_grad_norms(self.tbwriter, self.chunks_trained_on, self.grad_ems)
                log.per_chunk_adv_field_loss(self.tbwriter, self.chunks_trained_on, self.chunk_raw_adv_field_loss,)
                if "classif" in self.chunk_raw_loss_terms:
                    log.per_chunk_classifier_loss(self.tbwriter,self.chunks_trained_on, float(self.chunk_raw_loss_terms["classif"]))
                
                #Epoch Loss logging
                for key in self.chunk_raw_loss_terms.keys():
                    raw_v = self.chunk_raw_loss_terms[key]
                    self.epoch_raw_loss_terms[key] += float(raw_v)
                    if key not in batch_normed_loss_terms: continue
                    norm_v = self.chunk_normed_loss_terms[key]
                    self.epoch_normed_loss_terms[key] += float(norm_v)
                for field, loss in self.chunk_raw_adv_field_loss.items():
                    self.epoch_raw_adv_field_loss[field] +=float(loss) if torch .is_tensor(loss) else float(loss)

                self.sum_chunk_train_times += time.time() - t0_chunk
            #IN SCOPE EPOCH
            log.per_epoch_raw_loss(self.tbwriter, epoch, dict(self.epoch_raw_loss_terms))
            log.per_epoch_normed_loss(self.tbwriter, epoch, dict(self.epoch_normed_loss_terms))
            log.per_epoch_raw_adv_field_loss(self.tbwriter, epoch, dict(self.epoch_raw_adv_field_loss))
            if "classif" in self.epoch_raw_loss_terms:
                log.per_epoch_classifier_loss(self.tbwriter, epoch, float(self.epoch_raw_loss_terms["classif"]))
        #FINISHED TRAINING 
        log.log_metadata_influence(self.tbwriter, self.head_logit_l2_ems, global_step=epoch)
        self.tbwriter.close()
    

    def train_on_batch(self, expr_batch: Tensor, meta_batches: Dict[str, Tensor]):
        #current_batchs_size = expr_batch.size()[0] #TODO: Should be 0 or 1???

        #Establish Metadata Contexts
        
        batch_s_context = {f: meta_batches[f] for f in self.model.used_fields}
        batch_t_context, changed_fields, t_as_idxs = self.trans_gen_protocol(batch_s_context)
        
        #Protect against Frozen Classifier
        self.classifier.eval()
        for p in self.classifier.parameters():
            p.requires_grad_(False)
        self.model.train()
        for p in self.model.parameters():
            p.requires_grad_(True)

        #first_cycle
        st_cycle_out =  self.model(expr_batch, batch_s_context, batch_t_context)
        
        #second_cycle
        ts_cycle_out = self.model(st_cycle_out["X_st"], batch_t_context, batch_s_context) #Note ts_cycle_out["X_st"] is actually X_s -> t -> s
        
        #Log metadata infleucne
        if self.log_head_influence:
            with torch.no_grad():
                for head, logits in ts_cycle_out["head_logits"].items():
                    l2_batch= torch.linalg.vector_norm(logits, dim=1)
                    l2_scalar= float(l2_batch.mean().item())
                    self.head_logit_l2_ems[head].update(l2_scalar) #'base' or f'{field}' in fields_used

        #Gather Loss Terms 
        raw_adv_field_loss_terms = self.get_adversarial_loss(st_cycle_out["X_st"], t_as_idxs, changed_fields)
        raw_loss_terms = {   
            "recon": (raw_recon_loss_final := nn.functional.mse_loss(expr_batch, ts_cycle_out["X_st"], reduction="mean")),
            "integ": (raw_integration_loss := self.get_integration_loss(st_cycle_out["hidden_encodings"], ts_cycle_out["hidden_encodings"])),
            "adv_mean": (raw_adv_field_loss_terms["field_mean"]),
            "aggreg": (raw_adv_field_loss_terms["field_mean"] + raw_recon_loss_final + raw_integration_loss),
            "classif": 0.0
        }

        normed_loss_terms = self.norm_loss_terms(raw_loss_terms)

        #Generator Step
        self.generator_optimizer.zero_grad(set_to_none=True)
        normed_loss_terms["aggreg"].backward()
        #clip gradients
        grad_norms = float(nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)) #NOTE: max_norm hardcoded
        #Logging gradients 
        self.grad_ems.update(grad_norms)
       
        self.generator_optimizer.step()
        
        #Enforce non-negative weights by ReLU the weights after updated
        with torch.no_grad():
            for param in self.model.parameters():
                param.clamp_min_(0) #RELU

        #Classifier Step (supervised on Real Data)
        if self.should_train_classifier():
            self.classifier.train()
            for p in self.classifier.parameters():
                p.requires_grad_(True)

            logits_real = self.classifier(expr_batch.detach())
            s_as_idx = {f: torch.argmax(batch_s_context[f], dim=1).to(self.device, non_blocking=True).long() for f in self.model.used_fields} #conver onehot to int-idx for cross-entropy [0,0,1,0] -> 2, CE(logits, 2)
            classif_loss_term = [nn.functional.cross_entropy(logits_real[f], s_as_idx[f]) for f in self.model.used_fields]
            loss_c = torch.stack(classif_loss_term).mean()

            
            self.classifier_optimizer.zero_grad(set_to_none=True)
            loss_c.backward()
            self.classifier_optimizer.step()

            raw_loss_terms["classif"] = float(loss_c.item())
        return raw_loss_terms, normed_loss_terms, raw_adv_field_loss_terms

    def trans_gen_protocol(self, source_context):
        #Curreng protocol: Randomly change each field
        changed_fields = self.model.used_fields
        for field in changed_fields:
            device = source_context[field].device

            # Clone all fields to avoid in-place edits on caller's tensors
            target_context: Dict[str, Tensor] = {k: v.clone() for k, v in source_context.items()}

            t_as_idxs = {k: torch.argmax(target_context[k], dim =1)for k in self.model.used_fields} #Convert onehot to int-id's

            # Cardinality and batch size
            num_classes = int(self.field_specs_dict[field].get("cardinality", 0))
            assert num_classes > 0, f"Field '{field}' must have positive cardinality"

            batch_size = target_context[field].shape[0]

            # Current indices via argmax (works for valid one-hots; all-zero rows map to 0)
            old_idx = target_context[field].argmax(dim=1)

            #TODO: don't neccessarily need to have new !=old, new can ==old. 
            # Sample a shift in [1, num_classes-1] so new != old, then wrap
            shift = torch.randint(1, num_classes, (batch_size,), device=device)
            new_idx = (old_idx + shift) % num_classes  # guaranteed different from old_idx

            # One-hot -> float32
            target_context[field] = nn.functional.one_hot(new_idx, num_classes=num_classes).to(device=self.device, dtype=torch.float32)
            t_as_idxs[field] = new_idx

        return target_context, changed_fields, t_as_idxs
    
    #TODO: This is likely inverse. Ideally is target? Yes=low loss, No = high loss
    def get_adversarial_loss(self, x_st:Tensor,  t_as_idxs:Dict[str,Tensor], changed_fields =List[str])-> Tensor:
            self.classifier.eval()

            for p in self.classifier.parameters():
                p.requires_grad_(False)
            
            logits_trans = self.classifier(x_st)
            adv_terms = {}
            for f in changed_fields:
                adv_terms[f] = nn.functional.cross_entropy(logits_trans[f], t_as_idxs[f]) # CE adds -log to stop gradient explosion slightly better than SM(1-P(t))
            val_list = list(adv_terms.values())
            adv_terms["field_mean"] = torch.stack(val_list).mean() if val_list else x_st.new_zeros(())
            return adv_terms
    
    #TODO: average by field or use raw? Probably not, just do raw total for now. 
    def get_integration_loss(self, h1s:Dict[str, Tensor], h2s:Dict[str, Tensor]):
        total_integration_loss = 0.0
        for head in h1s:
            total_integration_loss += nn.functional.mse_loss(h1s[head], h2s[head])
        return total_integration_loss
    
    #TODO:  
    def norm_loss_terms(self, raw_terms):
        return raw_terms



    def should_train_classifier(self):
        return True
    
    #TODO: slight pretrain of classifier?
    def classifier_warmup():
        return 
        #
        # for 1 epoch
        #   CE(real_logits, unchanged meta)#