from dataclasses import dataclass
from typing import Optional, Tuple, Any, Dict, List
import torch

from transformers.file_utils import (
    add_code_sample_docstrings,
    add_end_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
    ModelOutput
)
from transformers.modeling_outputs import (
    Seq2SeqLMOutput
)

@dataclass
class MySeq2SeqLMOutput(Seq2SeqLMOutput):
    """
    Extra sequence-to-sequence language models outputs.
    Args:
        pairing_constraint (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indicates which left brackets are allowed to be paired in the next step.
            This is used to prevent generating invalid sequences.
        pairing (`torch.LongTensor` of shape `(batch_size, 1)`, *optional*):
            pairing left brackets for right brackets.
        linking (`torch.LongTensor` of shape `(batch_size, 1)`, *optional*):
            linking right brackets to right brackets.
        typing (`torch.LongTensor` of shape `(batch_size, 1)`, *optional*):
            typing right brackets.
    """
    pairing_constraint: Optional[List[torch.BoolTensor]] = None
    pairing: Optional[List[torch.LongTensor]] = None
    linking: Optional[List[torch.LongTensor]] = None
    typing: Optional[List[torch.LongTensor]] = None
