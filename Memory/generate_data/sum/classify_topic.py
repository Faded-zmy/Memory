import openai
import time
import tiktoken
import json
import os
from multiprocessing import Process
from tqdm import tqdm
import re

def openai_chatcompletion(prompt, api_id=0):
    api_key = [
    # "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",
    "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",
    # "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",
    # "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",
    # "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"
    ]
    openai.api_key = api_key[api_id]
    cn = 0
    enc = tiktoken.get_encoding("cl100k_base")
    # print(f"Tokens Of Prompt in {api_id}: {len(enc.encode(prompt))}!")
    if len(enc.encode(prompt)) > 7000:
        return None
    # max_token = 15000 - len(enc.encode(prompt))

    while True:
        output = ""
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k",
                # model="gpt-4",
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=1,
                # max_tokens=max_token,
                request_timeout = 120,
                timeout = 120
            )
            output = response['choices'][0]['message']['content']
            break
        except openai.error.OpenAIError as e:
            print(f"error {e} {api_id}")
            # logging.warning("Hit request rate limit; retrying...")
            #print("error cn", cn)
            # if e.json_body != None:
            #     if e.json_body['error']['type'] == 'invalid_request_error':
            #         max_token -= 500
            cn += 1
            if cn > 3:
                print(f"Already tried more than {cn} times!")
                break
            time.sleep(18)

    return output

def classify_prompt(topic):
    prompt = f"""Classify the topic "{topic}".
Requirement:
1. If the topic is about personal information or personal experience description or interpersonal relationship, fill in "Class" with "Personal Information", else fill in "Class" with "Discussion".
2. Classify it into one of the categories of the corresponding class.
3. Provide a score (between 0 and 10) for the degree of matching between the category and the topic.

Categories:
    Personal Information: ['Personal Experience', 'Personal Preferences', 'Personal achievements', 'Personal strengths and weaknesses', 'Interpersonal Relationship']
    Discussion: ['Business', 'Entertainment', 'Politics', 'Sports', 'World', 'LifeStyle', 'ScienceAndTechnology', 'Education', 'Military', 'RealEstate', 'Society', 'Health', 'Nature']

Output format:
    Class: xxx
    Category: xxx
    Score: xxx
"""
    return prompt

def PostProcess(topic, response):
    splitted_data = re.split(f"(Class|Category|Score):", response, flags=re.I)
    Class = splitted_data[2].strip()
    Category = splitted_data[4].strip()
    Score = splitted_data[6].strip()
    res = {'topic': topic, 'class': Class, 'category': Category, 'score': Score}
    return res

def generate_prompt_reply(json_path, output_path, split_id):
    topic_data = json.load(open(json_path, 'r'))['Dis']
    for td in topic_data:
        topic = td['T']
        prompt = classify_prompt(topic)
        # print('='*60)
        # print("PROMPT:\n", prompt)
        # print('='*60)
        reply = openai_chatcompletion(prompt)
        try:
            res = PostProcess(topic, reply)
            # print('='*60)
            # print("RES:\n", res)
            # print('='*60)
            f = open(output_path, 'a',  encoding='utf-8')
            json.dump(res, f, ensure_ascii=False)
            f.write('\n')
        except Exception as e:
            print("Error:", e)
            print("Reply:", reply)
            print("Topic:", topic)
            print('='*60)
            continue
    

def work(start, end, combinations, output_path, split_id):
    
    for i in tqdm(range(start, end), total=end-start):
        json_path = combinations[i]
        # print(json_path)
        if json_path.split('.')[-1] == 'json':
            # print(json_path)
            generate_prompt_reply(json_path=json_path,
                                                output_path=output_path, split_id=split_id)

if __name__ == "__main__":
    folders = ['/ai_jfs/mengying/data/sum/characters_dialogue_20230608_gpu1',
               '/ai_jfs/mengying/data/sum/characters_dialogue_20230608_gpu2',
               ]
   
    # topic = "Homer's Pet Lobster, Pinchy"
    # output_file = open('/ai_jfs/mengying/data/classify_topic/topic_classify_result_0808.jsonl', 'a', encoding='utf-8')
    output_path = '/ai_jfs/mengying/data/classify_topic/topic_classify_result_0808.jsonl'
    for root_path in folders:
        # root_path = '/ai_efs/lifeng/data/character/characters_dialogue/characters_dialogue_20230705_gpu2/'
        combinations = [root_path+'/'+file for file in os.listdir(root_path) if file.endswith('.json')]
        num_process = 10
        division = len(combinations) // num_process
        ranges = [i*division for i in range(num_process)]
        processes = []
        # output_dir = '/ai_jfs/mengying/data/sum/'+root_path.split('/')[-1]+'/'
        
        for i in range(num_process):
            # save_path = os.path.join(output_dir, "regen_roomchat_specific_news_"+str(i)+".json")
            if i == num_process - 1:
                start = ranges[i]
                end = len(combinations)
                p = Process(target=work, args=(start, end, combinations, output_path, i))
            else:
                start = ranges[i]
                end = ranges[i+1]
                p = Process(target=work, args=(start, end, combinations, output_path, i))
            p.start()
            processes.append(p)
            time.sleep(36)
        for p in processes:
            p.join()
   
    
        