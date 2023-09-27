from transformers import AutoTokenizer, AutoModel
import torch
from datasets import load_dataset
import pandas as pd
import json
import time
import os
import openai
import logging

def openai_chatcompletion2(prompt):
    cn = 0
    while True:
        output = ""
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=1.0,
            )
            output = response['choices'][0]['message']['content']
            break
        except openai.error.OpenAIError as e:
            logging.warning("Hit request rate limit; retrying...")
            #print("error cn", cn)
            cn += 1
            if cn == 2:
                break
            time.sleep(1)

    return output


def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]

def get_embeddings(text_list):
    encoded_input = tokenizer(
        text_list, padding=True, truncation=True, return_tensors="pt"
    )
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    model_output = model(**encoded_input)
    return cls_pooling(model_output)

def prompt_template(contexts,question):
    return f"""Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.\n{contexts}\nQuestion: {question}\nHelpful Answer:"""


#用hf的模型生成embedding库
##load聊天记录
test_dataset = load_dataset('text',data_files="/home/ec2-user/mengying/Memory/embedding/test.txt")
##load模型
model_ckpt = "/home/ec2-user/mengying/Memory/embedding/multi-qa-mpnet-base-dot-v1"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt)
device = torch.device("cuda")
model.to(device)
##生成embedding库
embeddings_dataset = test_dataset.map(
    lambda x: {"embeddings": get_embeddings(x["text"]).detach().cpu().numpy()[0]}
)
##将embedding库保存在本地
embeddings_dataset['train'].to_json('./embeddings_dataset.json')


#load embedding库 并search相关片段
##load 库
embeddings_dataset = load_dataset('json',data_files='./embeddings_dataset.json')['train']
##添加index
embeddings_dataset.add_faiss_index(column="embeddings")
##对question编码
question = "What museum did Member2 visit in Japan?"
question_embedding = get_embeddings([question]).cpu().detach().numpy()
##在库里search相关片段
scores, samples = embeddings_dataset.get_nearest_examples(
    "embeddings", question_embedding, k=3
)
##读结果
samples_df = pd.DataFrame.from_dict(samples)
samples_df["scores"] = scores
samples_df.sort_values("scores", ascending=False, inplace=True)
contexts = ''
for _, row in samples_df.iterrows():
    contexts = contexts+row['text']+'\n'
    print('text',row['text'])
    print('score',row['scores'])
    print()



#将问题和contexts塞进模型(chatgpt)得到答案
os.environ["OPENAI_API_KEY"] = "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"
prompt = prompt_template(contexts,question)
print('prompt',prompt)
result = openai_chatcompletion2(prompt)
print('result',result)