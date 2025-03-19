import torch
import torch.nn.functional as F

# Change it with respect to the original model
from config import LlamaConfig
from llama import load_pretrained
from tokenizer import Tokenizer

class LlamaZeroShotClassifier(torch.nn.Module):
    def __init__(self, config: LlamaConfig, tokenizer: Tokenizer, label_names: list[str]):
        super(LlamaZeroShotClassifier, self).__init__()
        self.num_labels = config.num_labels
        self.llama = load_pretrained(config.pretrained_model_path)
        # Zero-shot classification does not require updating llama parameters.
        for param in self.llama.parameters():
            param.requires_grad = False
        assert len(label_names) == self.num_labels
        self.tokenizer = tokenizer
        self.label_name_ids = [tokenizer.encode(label, bos=False, eos=False) for label in label_names]

    def forward(self, input_ids):
        # Compute the completion probability of each label string
        logits, _ = self.llama(input_ids)
        log_probabilities = F.log_softmax(logits, dim=-1)
        label_probabilities = torch.zeros((log_probabilities.shape[0], self.num_labels), device=log_probabilities.device)
        for i, label_token_ids in enumerate(self.label_name_ids):
            total_log_prob = torch.sum(log_probabilities[:, :, label_token_ids], axis=-1)
            label_probabilities[:, i] = total_log_prob[:, 0]
        return label_probabilities

class LlamaEmbeddingClassifier(torch.nn.Module):
    def __init__(self, config):
        super(LlamaEmbeddingClassifier, self).__init__()
        self.num_labels = config.num_labels
        self.llama = load_pretrained(config.pretrained_model_path)
        # If we use pretrain mode, we freeze Llama parameters.
        for param in self.llama.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True

        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier_head = torch.nn.Linear(config.dim, self.num_labels)  # Sử dụng config.dim

    def forward(self, input_ids):
        '''
        1) Find the hidden state after the final token of the input sequence
        2) Apply dropout (self.dropout) to the hidden state at training time to mitigate
            overfitting.
        3) Pass this through the classifier head (self.classifier_head), which will return
            logits (unnormalized probabilities) over all classes.
        4) Take the log-softmax of the logits and return log-probabilities over all classes.
        '''
        # 1) Obtain the model's output for all tokens
        outputs = self.llama(input_ids)
        # print(f"TypeType of outputs: {type(outputs)}")  # In shape of outputs
        logits, hidden_states = outputs  # Shape: (batch_size, sequence_length, hidden_size)
        # print(f"Shape of hidden_states: {hidden_states.shape}")
        # 2) Select the output of the final token
        final_hidden_state = hidden_states[:, -1, :]  # Shape: (batch_size, hidden_size)
        # print(f"Shape of final_hidden_state: {final_hidden_state.shape}")
        # 3) Apply dropout only during training
        if self.training:
            final_hidden_state = self.dropout(final_hidden_state)
        # 4) Obtain logits from the classifier head
        logits = self.classifier_head(final_hidden_state)  # Shape: (batch_size, num_labels)
        # 5) Compute log-probabilities using log-softmax
        log_probabilities = F.log_softmax(logits, dim=-1)  # Shape: (batch_size, num_labels)

        return log_probabilities