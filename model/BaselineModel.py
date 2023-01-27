import torch.nn as nn
import einops as ein
from transformers import AutoModel, LayoutLMv2PreTrainedModel

class BaselineModel(LayoutLMv2PreTrainedModel):
    """_summary_
    베이스라인 모델입니다.
    """
    def __init__(self, model_name, num_labels, dropout_rate):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.num_labels = num_labels
        self.model_name = model_name
        
        self.model = AutoModel.from_pretrained(self.model_name)

        self.qa_outputs = nn.Sequential(
            # nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.model.config.hidden_size, self.num_labels)
        )

    def forward(self,
        input_ids=None,
        bbox=None,
        image=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        start_positions=None,
        end_positions=None
    ):
        seq_length = input_ids.size()[1]
        sequence_output = self.model(
            input_ids=input_ids,
            bbox=bbox,
            image=image,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )[0][:, :seq_length] # last_hidden_state

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = ein.rearrange(start_logits, 'batch seq 1 -> batch seq')
        end_logits = ein.rearrange(end_logits, 'batch seq 1 -> batch seq')

        return start_logits, end_logits