from transformers import AutoTokenizer, AutoModel
import torch
from datasets import load_dataset
import pandas as pd
import json
import time

def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]

def get_embeddings(text_list):
    encoded_input = tokenizer(
        text_list, padding=True, truncation=True, return_tensors="pt"
    )
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    model_output = model(**encoded_input)
    return cls_pooling(model_output)


test_dataset = load_dataset('text',data_files="./test.txt")
print('test_dataset',test_dataset["train"])


model_ckpt = "./multi-qa-mpnet-base-dot-v1"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt)
device = torch.device("cuda")
model.to(device)

# embedding = get_embeddings('Member4 says:" It was beautiful! We visited a lot of shrines and temples and even saw some geishas.')
# print(embedding.shape)
embeddings_dataset = test_dataset.map(
    lambda x: {"embeddings": get_embeddings(x["text"]).detach().cpu().numpy()[0]}
)

print('embeddings_dataset',embeddings_dataset['train'])
# embeddings_dataset['train'].add_faiss_index(column="embeddings")

embeddings_dataset['train'].to_json('./embeddings_dataset.json')
embeddings_dataset_zmy = load_dataset('json',data_files='./embeddings_dataset.json')['train']
t1 = time.time()
embeddings_dataset_zmy.add_faiss_index(column="embeddings")
t2 = time.time()
print('add index need',t2-t1)
# f = open('./embeddings_dataset.json','w')

# f.write(json.dumps(embeddings_dataset['train'],indent=3))

question = "What museum did Member2 visit in Japan?"
question_embedding = get_embeddings([question]).cpu().detach().numpy()
question_embedding.shape

scores, samples = embeddings_dataset_zmy.get_nearest_examples(
    "embeddings", question_embedding, k=3
)

samples_df = pd.DataFrame.from_dict(samples)
samples_df["scores"] = scores
samples_df.sort_values("scores", ascending=False, inplace=True)

for _, row in samples_df.iterrows():
    print('text',row['text'])
    print('score',row['scores'])
    print()

# print('result',scores,samples)
# embeddings_dataset = comments_dataset.map(
#     lambda x: {"embeddings": get_embeddings(x["text"]).detach().cpu().numpy()[0]}
# )
