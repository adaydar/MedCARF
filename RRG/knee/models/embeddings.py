import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import numpy as np

class BERTModel(nn.Module):
    def __init__(self, bert_type, project_dim=256):  # Set project_dim to 256
        super(BERTModel, self).__init__()
        self.model = AutoModel.from_pretrained(bert_type, output_hidden_states=True, trust_remote_code=True)
        self.project_head = nn.Sequential(
            nn.Linear(768, project_dim),  
            nn.LayerNorm(project_dim),
            nn.GELU()
        )
        for param in self.model.parameters():
            param.requires_grad = False  

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
        last_hidden_states = torch.stack([output.hidden_states[1], output.hidden_states[2], output.hidden_states[-1]])
        embed = last_hidden_states.permute(1, 0, 2, 3).mean(2).mean(1)  
        embed = self.project_head(embed)  
        return embed  # Shape: [1, 256]

class getEMBD(nn.Module):
    def __init__(self):
        super(getEMBD, self).__init__()
        self.text_encoder = BERTModel("dmis-lab/biobert-base-cased-v1.1")  
        self.bert_tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

        self.disease_description = [
            "No radiographic evidence of osteoarthritis.Normal joint space and no osteophytes.",
            "Small osteophytes (bone spurs) present.No significant joint space narrowing.",
            "Definite osteophyte is present.Possible narrowing of the joint space.No significant deformity or sclerosis.",
            "Multiple osteophytes is seen.Moderate joint space narrowing.Some sclerosis and possible deformity of the bone.",
            "Large osteophytes is seen.Severe joint space narrowing.Subchondral sclerosis and cyst formation.Significant bony deformity."
        ]

        self.precomputed_embeddings = self._compute_disease_embeddings()
    
    def _compute_disease_embeddings(self):
        disease_embeddings = {}
        for idx, description in enumerate(self.disease_description):
            tokenized_report = self.bert_tokenizer(
                description,
                padding="max_length",
                max_length=24,
                truncation=True,
                return_tensors="pt"
            )

            input_ids = tokenized_report['input_ids']
            attention_mask = tokenized_report['attention_mask']

            with torch.no_grad():
                embedding = self.text_encoder(input_ids, attention_mask)  

            disease_embeddings[idx] = embedding

        return disease_embeddings  

    def forward(self, disease_labels_batch):
        batch_size = 1  
        embeddings_list = []  
        
        for i in range(batch_size):
            active_labels = list(np.where(disease_labels_batch[i].cpu().numpy() == 1)[0])  

            if not active_labels:
                active_labels = [0]  

            selected_embeddings = [self.precomputed_embeddings[idx] for idx in active_labels]

            if len(selected_embeddings) > 1:
                final_embedding = torch.mean(torch.cat(selected_embeddings, dim=0), dim=0, keepdim=True)  
            else:
                final_embedding = selected_embeddings[0]  
            
            # Repeat the embedding to create a 128x256 tensor
            final_embedding = final_embedding.repeat(128, 1)  # Shape: [128, 256]

            embeddings_list.append(final_embedding)  

        return embeddings_list  # Return the list of tensors

# model = getEMBD()

# batch_size = 8
# num_classes = 5
# random_indices = torch.randint(0, num_classes, (batch_size,))  
# one_hot_input = torch.zeros(batch_size, num_classes)
# one_hot_input[torch.arange(batch_size), random_indices] = 1  
# print(one_hot_input)
# output_list = model(one_hot_input)  # Get the list of tensors

# # Print the shapes of the tensors in the list
# for i, tensor in enumerate(output_list):
#     print(f"Tensor {i + 1} shape: {tensor.shape}")  # Expected: [128, 256]