import torch
from config import LlamaConfig
from classifier import LlamaEmbeddingClassifier
from tokenizer import Tokenizer

# Khởi tạo cấu hình
config = LlamaConfig(
    pretrained_model_path="stories42M.pt",  # Đường dẫn đến pretrained weights
    num_labels=2,  # Số lượng lớp phân loại
    hidden_dropout_prob=0.1,  # Dropout probability
    option="finetune"  # Chế độ: 'pretrain' hoặc 'finetune'
)

# Khởi tạo tokenizer (giả sử tokenizer đã được triển khai)
tokenizer = Tokenizer()

# Khởi tạo mô hình
model = LlamaEmbeddingClassifier(config)

# Chuẩn bị dữ liệu đầu vào (ví dụ)
input_ids = torch.tensor([[1, 2, 3, 4, 5]])  # Chuỗi token hóa

# Chạy mô hình
log_probabilities = model(input_ids)
print("Log probabilities:", log_probabilities)