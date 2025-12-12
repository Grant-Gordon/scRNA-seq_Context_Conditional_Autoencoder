#iny_outy_datalaoder.py
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as func
from scipy import sparse
import glob
import pickle
import os
from typing import Tuple, Dict, Any
import pandas as pd

#Outy dataloader
class ChunksDataset(Dataset):
    def __init__(self, data_dir_path:str, meta_glob_pattern:str, gene_expr_glob_pattern:str):

        meta_glob_path = os.path.join(data_dir_path, meta_glob_pattern)
        gene_expr_glob_path = os.path.join(data_dir_path, gene_expr_glob_pattern)

        meta_files = sorted(glob.glob(meta_glob_path))
        gene_expr_files = sorted(glob.glob(gene_expr_glob_path))
        assert len(meta_files) == len(gene_expr_files), " Mismatched counts of files. Meta-expr files should Align"

        self.chunks = list(zip(gene_expr_files, meta_files))


    def __getitem__(self, index):
        expr_path, meta_path = self.chunks[index]
 
        chunk_expr = sparse.load_npz(expr_path)

        with open(meta_path, 'rb') as f:
            chunk_meta = pickle.load(f)


        return chunk_expr, chunk_meta
    
    def __len__(self):
        return len(self.chunks)


#iny dataloaders 
class SingleChunkDataset(Dataset):
    def __init__(self, chunk, field_specs, used_fields, field_value_map):
        self.field_specs = field_specs
        self.used_fields = used_fields        
        self.gene_expr_csr = chunk[0]
        self.samples_in_chunk = self.gene_expr_csr.shape[0]
        
        self. meta = chunk[1]
        self.field_value_map = field_value_map


    def __getitem__(self, index):
        row = self.gene_expr_csr[index]
        expr = torch.tensor(row.toarray().flatten(), dtype=torch.float32)
        
        meta_row=self.meta.iloc[index]
        meta_onehots = self.meta_raw_to_onehot(meta_row) 
        # meta_onehots = {} #NOTE: USED FOR DEBUG ONLY 
      
        return expr, meta_onehots

    def __len__(self):
        return self.samples_in_chunk


    def meta_raw_to_onehot(self, meta_row_raw):

        out = {}
        for field in self.used_fields:
            field_cardinality = self.field_specs[field]["cardinality"]
            value_to_idx = self.field_value_map.get(field, {})
            raw_val = meta_row_raw.get(field, None)

            idx = value_to_idx.get(raw_val, None)
            one_hot = func.one_hot(torch.tensor(idx, dtype=torch.long), num_classes=field_cardinality).to(torch.float32)

            out[field] = one_hot

        return out
# #Trainier psuedo code
#  heads = module dict... yada yada 
# outerlaoder = DataLoader(ChunksDataset(DATA_DIR, GLOB_PATTERNS), num_workers=2, prefetch_factor=1)
# for chunk in chunklaoder:
#     innerloader = DataLoader(SingleChunkDataset(chunk, JSONS), num_workers=4, prefetch_factor=2))
#     for expr, meta_onehots in innerlaoder:
#         heads["expr"].forward(expr)
#         for field in self.used_fields:
#             heads[field].foward(meta_onehots[field])