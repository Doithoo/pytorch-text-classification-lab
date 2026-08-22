import torch
from torch import nn

from text_classifier.models import EmbeddingBagClassifier

model = EmbeddingBagClassifier(vocab_size=20, num_classes=4, embedding_dim=8)
input_ids = torch.tensor([[2, 5, 3], [2, 6, 7]])
attention_mask = torch.ones_like(input_ids)
labels = torch.tensor([0, 2])
logits = model(input_ids, attention_mask)
loss = nn.CrossEntropyLoss()(logits, labels)
loss.backward()
print("logits:", logits.shape)
print("loss:", float(loss))
print("minimal forward/backward OK")
