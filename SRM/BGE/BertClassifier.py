from torch import nn
from transformers import BertModel
from transformers import AutoTokenizer, AutoModel

class BertClassifier(nn.Module):
    def __init__(self, dropout=0.5):
        super(BertClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained('/mmu_nlp_hdd/xiayu12/LIBER/llm/longformer-base-4096',  trust_remote_code=True)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 2)

    def forward(self, input_id, mask):
        output = self.bert(input_ids= input_id, attention_mask=mask)
        pooled_output = output.pooler_output
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        return linear_output


# model = AutoModel.from_pretrained('/mmu_nlp_hdd/xiayu12/LIBER/llm/longformer-base-4096',  trust_remote_code=True)
# model = BertClassifier(0.2)
# model.train()
# tokenizer = AutoTokenizer.from_pretrained('/mmu_nlp_hdd/xiayu12/LIBER/llm/longformer-base-4096',  trust_remote_code=True)
# text = '你是谁[SEP]喜欢你'
# inputs = tokenizer([text]*10, padding='max_length', truncation=True, max_length=4096, return_tensors="pt",
#                               return_attention_mask=True)
# # print(model(**inputs))

# output = model(inputs['input_ids'],inputs['attention_mask'])
# print(output.shape)