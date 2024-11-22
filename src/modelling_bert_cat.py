from transformers import (
    PreTrainedModel,
    PretrainedConfig,
    PreTrainedTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
)
import torch
import torch.nn.functional as F
from rankers.modelling.base import Ranker


class CatConfig(PretrainedConfig):
    model_type = "Cat"


class Cat(Ranker):
    """Wrapper for Cat Model

    Parameters
    ----------
    model : PreTrainedModel
        the underlying HF model
    tokenizer : PreTrainedTokenizer
        the tokenizer for the model
    config : AutoConfig
        the configuration for the model
    """

    model_type = "Cat"
    architecture_class = AutoModelForSequenceClassification
    config_class = CatConfig
    transformer_class = None

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: AutoConfig,
    ):
        super().__init__(model, tokenizer, config)
        from rankers.pyterrier.cat import CatTransformer

        self.transformer_class = CatTransformer

    def prepare_outputs(self, logits):
        """Prepare outputs"""
        return F.log_softmax(logits, dim=-1)[:, 1]

    def forward(self, **inputs):
        """Compute the loss given (pairs, labels)"""
        sequences = {k: v.to(self.model.device) for k, v in inputs.items()}
        logits = self.model(**sequences).logits
        pred = self.prepare_outputs(logits)
        return pred  # B x Group x 1
