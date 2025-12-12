#refactor_core.py
import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Tuple

class CAE(nn.Module):
    def __init__(self, input_dim:int, latent_dim:int, field_specs:Dict[str, Dict[str, int]]):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.field_specs = field_specs
        #Encoder- tied-weihghts for dec
        self.base_encoder = nn.Linear(input_dim, latent_dim, bias=False)
        #TODO: implement Moore-Penrose iterative updates of a He initialization 
        
        self.used_fields = [f for f, spec in field_specs.items() if spec.get("using", False)]

        #Field Shared Enc/Dex + per-context Heads
        self.shared_meta_encoders = nn.ModuleDict()
        self.shared_meta_decoders = nn.ModuleDict()
        self.field_context_head_pool = nn.ModuleDict()
        for field in self.used_fields:
            #shraed Enc/Dec's
            card = int(field_specs[field].get("cardinality", 0))
            assert card> 0, f"field: {field} must have cardinality greater than 0"
            self.shared_meta_encoders[field] = nn.Linear(input_dim, latent_dim, bias=False) 
            self.shared_meta_decoders[field] = nn.Linear(latent_dim, input_dim, bias=False) 
        
            #Per-context FFN head.
            self.field_context_head_pool[field] = nn.ModuleDict()
            for context in range(card):
                self.field_context_head_pool[field][str(context)] = nn.Linear(latent_dim, latent_dim, bias=False)
            
        #init weights for all heads
        CAE._weight_init(self.base_encoder)
        for head in self.shared_meta_encoders.values():
            CAE._weight_init(head)
        for head in self.shared_meta_decoders.values():
            CAE._weight_init(head)
        for field_pool in self.field_context_head_pool.values():
            for context_head in field_pool.values():
                CAE._weight_init(context_head)



    
    def forward(self, expr:Tensor, source_context: Dict[str, Tensor], target_context:Dict[str, Tensor]) -> Dict[str, Tensor]:
        hidden_encodings = {} # Cache these for integration loss terms later 
        head_logits = {} # Used to track the influence of heads Base & all fields 

        #Base encoder. Simple X W^TW
        hidden_encodings["base"] = self.base_encoder(expr)   #hx = X W^T
        head_logits["base"] = torch.matmul(hidden_encodings["base"], self.base_encoder.weight) # (XW^T)W
        for field in self.used_fields:
            #Enc With Source Metadata
            shared_encoding = self.shared_meta_encoders[field](expr)
            hidden_encodings[field] = self._apply_heads_by_context(x=shared_encoding, context=source_context[field], head_pool=self.field_context_head_pool[field]) #h_field = X * C_shared_enc * Cs_FFN

            #Dec with Target Metadata
            decoded_context =  self._apply_heads_by_context(x=hidden_encodings[field], context=target_context[field], head_pool=self.field_context_head_pool[field])
            head_logits[field] = self.shared_meta_decoders[field](decoded_context) * 0.0159 #NOTE: constant so that X_st ~= 2 *base_dec + 1(total_field_dec). (prelimary runs showed each meta_head_dec was ~3.5X base_dec)
        
        #Combine Head Outputs and return 
        X_st = torch.stack(list(head_logits.values()), dim=0,).sum(dim=0)
        return {
            "X_st": X_st,   #Trans genreated transcriptome from contest s->t
            "hidden_encodings": hidden_encodings, #Cache hidden layers for integration loss later
            "head_logits": head_logits #Cache Base + Metadata logits for tracking of metadata influence on final epoch
        }


    #TODO: Try columns not rows. (see if column scaling gives better init) 
    @staticmethod
    def _weight_init(module: nn.Module ):
        with torch.no_grad():
            # #set weights to uniform dist
            # W = module.weight
            # W.uniform_(0.0, 1.0)
            # #Sum rows (for denominator) calmped for divide-by-zero gaurd
            # row_norms = torch.linalg.vector_norm(W, dim=1, keepdim=True).clamp_min(1e-12) #Dim = 1 for rows dim=0 for cols 
            # #normalize Row In place by dividing by row_sums
            # W.div_(row_norms)
            # #DEBUG "Do all row L2s look like 1?"
            # assert torch.allclose(torch.linalg.vector_norm(W, dim=1), torch.ones(W.size(0)), atol=1e-6)
          
            # W = shape (out_features, in_features)
            #dim=1 -> norm across rows (in_features)
            #dim=0 -> norm across cols (out_features)
            #set weights to uniform dist
            W = module.weight
            W.uniform_(0.0, 1.0)
            # Column L2 norms (denominator), with guard against divide-by-zero
            col_norms = torch.linalg.vector_norm(W, dim=0, keepdim=True).clamp_min(1e-12) #Dim = 1 for rows dim=0 for cols, #keepdim=True allows for broadcast on div
            #normalize Row In place by dividing by row_sums
            W.div_(col_norms)
            #DEBUG "Do all Cols L2s look like 1?"
            assert torch.allclose(torch.linalg.vector_norm(W, dim=0), torch.ones(W.size(1)), atol=1e-6)

    def _apply_heads_by_context(self, x: Tensor, context: Tensor, head_pool:nn.ModuleDict):

        context_ids = context.argmax(dim=1).to(device=x.device, dtype=torch.long)  # Convert one-hot to indices)

        # Get unique context IDs in this batch
        unique_ctx = context_ids.unique()
        out= None

        for context_val in unique_ctx:
            # Indices of samples with this context
            idx = (context_ids == context_val).nonzero(as_tuple=True)[0]

            #looup head
            key = str(int(context_val.item()))
            head = head_pool[key]

            #sub-batch input for this context 
            sub_in = x.index_select(dim=0, index=idx)
            sub_out = head(sub_in)

            # Lazily allocate output buffer with matching trailing shape
            if out is None:
                out = sub_out.new_zeros((x.size(0),) + sub_out.shape[1:])

            out.index_copy_(dim=0, index=idx, source=sub_out)

        return out