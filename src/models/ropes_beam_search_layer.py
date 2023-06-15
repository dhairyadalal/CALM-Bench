from typing import *
from transformers import PretrainedConfig
from transformers.modeling_utils import get_parameter_dtype
import torch
import torch.nn as nn

class PoolerStartLogits(nn.Module):
    
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, 1)

    def forward(
        self, hidden_states: torch.FloatTensor, p_mask: Optional[torch.FloatTensor] = None
    ) -> torch.FloatTensor:
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, seq_len, hidden_size)`):
                The final hidden states of the model.
            p_mask (`torch.FloatTensor` of shape `(batch_size, seq_len)`, *optional*):
                Mask for tokens at invalid position, such as query and special symbols (PAD, SEP, CLS). 1.0 means token
                should be masked.
        Returns:
            `torch.FloatTensor`: The start logits for SQuAD.
        """
        x = self.dense(hidden_states).squeeze(-1)

        if p_mask is not None:
            if get_parameter_dtype(self) == torch.float16:
                x = x * (1 - p_mask) - 65500 * p_mask
            else:
                x = x * (1 - p_mask) - 1e30 * p_mask
        return x


class PoolerEndLogits(nn.Module):
    
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.dense_0 = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.activation = nn.Tanh()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dense_1 = nn.Linear(config.hidden_size, 1)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        start_states: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        p_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, seq_len, hidden_size)`):
                The final hidden states of the model.
            start_states (`torch.FloatTensor` of shape `(batch_size, seq_len, hidden_size)`, *optional*):
                The hidden states of the first tokens for the labeled span.
            start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                The position of the first token for the labeled span.
            p_mask (`torch.FloatTensor` of shape `(batch_size, seq_len)`, *optional*):
                Mask for tokens at invalid position, such as query and special symbols (PAD, SEP, CLS). 1.0 means token
                should be masked.
        <Tip>
        One of `start_states` or `start_positions` should be not `None`. If both are set, `start_positions` overrides
        `start_states`.
        </Tip>
        Returns:
            `torch.FloatTensor`: The end logits for SQuAD.
        """
        assert (
            start_states is not None or start_positions is not None
        ), "One of start_states, start_positions should be not None"
        if start_positions is not None:
            slen, hsz = hidden_states.shape[-2:]
            start_positions = start_positions[:, None, None].expand(-1, -1, hsz)  # shape (bsz, 1, hsz)
            start_states = hidden_states.gather(-2, start_positions)  # shape (bsz, 1, hsz)
            start_states = start_states.expand(-1, slen, -1)  # shape (bsz, slen, hsz)

        x = self.dense_0(torch.cat([hidden_states, start_states], dim=-1))
        x = self.activation(x)
        x = self.LayerNorm(x)
        x = self.dense_1(x).squeeze(-1)

        if p_mask is not None:
            if get_parameter_dtype(self) == torch.float16:
                x = x * (1 - p_mask) - 65500 * p_mask
            else:
                x = x * (1 - p_mask) - 1e30 * p_mask

        return x
    
    
class PoolerAnswerClass(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense_0 = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.activation = nn.Tanh()
        self.dense_1 = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        start_states: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None    
    ) -> torch.FloatTensor:
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, seq_len, hidden_size)`):
                The final hidden states of the model.
            start_states (`torch.FloatTensor` of shape `(batch_size, seq_len, hidden_size)`, *optional*):
                The hidden states of the first tokens for the labeled span.
            start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                The position of the first token for the labeled span.
            cls_index (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Position of the CLS token for each sentence in the batch. If `None`, takes the last token.
        <Tip>
        One of `start_states` or `start_positions` should be not `None`. If both are set, `start_positions` overrides
        `start_states`.
        </Tip>
        """
        # No dependency on end_feature so that we can obtain one single `cls_logits` for each sample.
        hsz = hidden_states.shape[-1]
        assert (
            start_states is not None or start_positions is not None
        ), "One of start_states, start_positions should be not None"
        if start_positions is not None:
            start_positions = start_positions[:, None, None].expand(-1, -1, hsz)  # shape (bsz, 1, hsz)
            start_states = hidden_states.gather(-2, start_positions).squeeze(-2)  # shape (bsz, hsz)

        if cls_index is not None:
            cls_index = cls_index[:, None, None].expand(-1, -1, hsz)  # shape (bsz, 1, hsz)
            cls_token_state = hidden_states.gather(-2, cls_index).squeeze(-2)  # shape (bsz, hsz)
        else:
            cls_token_state = hidden_states[:, -1, :]  # shape (bsz, hsz)

        x = self.dense_0(torch.cat([start_states, cls_token_state], dim=-1))
        x = self.activation(x)
        x = self.dense_1(x).squeeze(-1)

        return x

class RopesBeamTaskLayer(nn.Module):  
    """ 
    adapted XLNET / SquadHead 
    source: https://github.com/huggingface/transformers/blob/v4.18.0/src/transformers/modeling_utils.py
    """
    
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        
        self.start_n_top = 5
        self.end_n_top = 5
        
        self.start_logits = PoolerStartLogits(config)
        self.end_logits = PoolerEndLogits(config)
        
    def forward(self, 
            hidden_states: torch.FloatTensor,
            start_positions: Optional[torch.LongTensor] = None,
            end_positions: Optional[torch.LongTensor] = None,
            p_mask: Optional[torch.FloatTensor] = None
        ) -> dict: 
        
        start_logits = self.start_logits(hidden_states, p_mask=p_mask)
        
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, let's remove the dimension added by batch splitting
            for x in (start_positions, end_positions):
                if x is not None and x.dim() > 1:
                    x.squeeze_(-1)

            # during training, compute the end logits based on the ground truth of the start position
            end_logits = self.end_logits(hidden_states, start_positions=start_positions, p_mask=p_mask)

            loss_fct = nn.CrossEntropyLoss()
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

            return {"loss": total_loss}

        else:
            # during inference, compute the end logits based on beam search
            bsz, slen, hsz = hidden_states.size()
            start_log_probs = nn.functional.softmax(start_logits, dim=-1)  # shape (bsz, slen)

            start_top_log_probs, start_top_index = torch.topk(
                start_log_probs, self.start_n_top, dim=-1
            )  # shape (bsz, start_n_top)
            start_top_index_exp = start_top_index.unsqueeze(-1).expand(-1, -1, hsz)  # shape (bsz, start_n_top, hsz)
            start_states = torch.gather(hidden_states, -2, start_top_index_exp)  # shape (bsz, start_n_top, hsz)
            start_states = start_states.unsqueeze(1).expand(-1, slen, -1, -1)  # shape (bsz, slen, start_n_top, hsz)

            hidden_states_expanded = hidden_states.unsqueeze(2).expand_as(
                start_states
            )  # shape (bsz, slen, start_n_top, hsz)
            p_mask = p_mask.unsqueeze(-1) if p_mask is not None else None
            end_logits = self.end_logits(hidden_states_expanded, start_states=start_states, p_mask=p_mask)
            end_log_probs = nn.functional.softmax(end_logits, dim=1)  # shape (bsz, slen, start_n_top)

            end_top_log_probs, end_top_index = torch.topk(
                end_log_probs, self.end_n_top, dim=1
            )  # shape (bsz, end_n_top, start_n_top)
            end_top_log_probs = end_top_log_probs.view(-1, self.start_n_top * self.end_n_top)
            end_top_index = end_top_index.view(-1, self.start_n_top * self.end_n_top)

            start_states = torch.einsum("blh,bl->bh", hidden_states, start_log_probs)
            
            return {
                        "start_top_log_probs": start_top_log_probs,
                        "start_top_index": start_top_index,
                        "end_top_log_probs": end_top_log_probs,
                        "end_top_index": end_top_index
                   }