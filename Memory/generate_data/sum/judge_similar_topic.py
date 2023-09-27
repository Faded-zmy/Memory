import random
import openai
import logging
import time
import signal
import tiktoken
from multiprocessing import Process
import numpy as np
import pandas as pd
import os
import utils
import json
import time
import json
import os
import random
import re
import string
import multiprocessing
import itertools
from multiprocessing import Process
from datetime import datetime

import utils
import re
import pandas as pd
import logging
from tqdm import tqdm
import argparse


def openai_chatcompletion(prompt, api_id):
    api_key = [
    "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",
    "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",
    "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",
    "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",
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
            logging.warning("Hit request rate limit; retrying...")
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


def load_prompt(topics_ls):
    topics = ''
    for i,t in enumerate(topics_ls):
        topics += str(i+1)+'. '+t+'\n'

    prompt = f"""
    Please divide the topics below.
    Requirement:
    1. According to the topic, if they are talking about the same thing, group them together.
    2. Divide topics by event ignore the person in it.

    Topics:
    [Start of the topics]
    {topics}
    [End of the topics]

    The output format is as follows:
    Group1: xxx
        Topics: xxx
    Group2: xxx
        Topics: xxx
    """
    return prompt

def PostProcess(result):
    result = result.strip().replace('Main_Topic: ', '')
    return result


def generate_prompt_reply(topics_group, log, api_id):
    
    prompt = load_prompt(topics_group)
    print('-'*60)
    print(prompt)
    print('-'*60)
    log.info("Prompt is: \n {}".format(prompt))
    result = openai_chatcompletion(prompt,api_id)
    print('-'*60)
    print(result)

    log.info("output result: {}".format(result))
    res = {'org_topics': topics_group, 'result': result}
    return res
    




# def generate_instruction_following_data(
#         topics_group,
#         split_id=0,
#         api_id=0
# ):
    # log = logging.getLogger('mengying.zhou')
    # log.setLevel('INFO')
    # date = str(datetime.now().date())
    # if not os.path.exists('/ai_jfs/mengying/data/sum/log/log1/' + date):
    #     os.mkdir('/ai_jfs/mengying/data/sum/log/log1/' + date)
    # file_handler = logging.FileHandler('/ai_jfs/mengying/data/sum/log/log1/' + date + '/generate_characters_' + str(split_id) + '.log', encoding='utf-8')
    # file_handler.setLevel('INFO')
    # fmt_str = '%(asctime)s %(thread)d %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'
    # formatter = logging.Formatter(fmt_str)
    # file_handler.setFormatter(formatter)
    # log.addHandler(file_handler)

#     # pre prompts
    
#     try:
#         generate_prompt_reply(topics_group, split_id, log, api_id)
#     except Exception as e:
#         log.error("topics_group: {}, Error: {}".format(topics_group, e))
#         # print("Json_path: {}, Error: {}".format(json_path, e))


def work(start, end, topics, output_dir, split_id, api_id):
    log = logging.getLogger('mengying.zhou')
    log.setLevel('INFO')
    date = str(datetime.now().date())
    if not os.path.exists('/ai_jfs/mengying/data/sum/log/log1/' + date):
        os.mkdir('/ai_jfs/mengying/data/sum/log/log1/' + date)
    file_handler = logging.FileHandler('/ai_jfs/mengying/data/sum/log/log1/' + date + '/generate_characters_' + str(split_id) + '.log', encoding='utf-8')
    file_handler.setLevel('INFO')
    fmt_str = '%(asctime)s %(thread)d %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'
    formatter = logging.Formatter(fmt_str)
    file_handler.setFormatter(formatter)
    log.addHandler(file_handler)

    right_end = start
    res = []
    while right_end < end:
        num = random.randint(20,50)
        topics_group = topics[right_end:min(end,right_end+num)]
        right_end += num
        res_one = generate_prompt_reply(topics_group=topics_group, log=log, api_id=api_id)
        res.append(res_one)
    json.dump(res, open(output_dir+str(split_id)+'.json', 'w'), ensure_ascii=False, indent=4)



def main_process(folders, api_id):
    # with open('roomchat_person_selected_news.json', 'r', encoding='utf-8') as f:
    #     combinations = utils.jload(f)
    # print(combinations)
    # folders = ['/ai_jfs/mengying/data/sum/sum_conv/characters_dialogue_20230608_gpu1']
    for root_path in folders:
        # root_path = '/ai_efs/lifeng/data/character/characters_dialogue/characters_dialogue_20230705_gpu2/'
        # print(root_path)
        # combinations = [root_path+'/'+file for file in os.listdir(root_path) if file.endswith('.json')]
        topics = []
        for file in os.listdir(root_path):
            if file.endswith('.json'):
                data = json.load(open(root_path+'/'+file, 'r'))
                for d in data:
                    topics.append(d['main_topic'])

        random.shuffle(topics)
        num_process = 1
        division = len(topics) // num_process
        ranges = [i*division for i in range(num_process)]
        processes = []
        output_dir = '/ai_jfs/mengying/data/sum/topic_group'+root_path.split('/')[-1]+'/'
        for i in range(num_process):
            # save_path = os.path.join(output_dir, "regen_roomchat_specific_news_"+str(i)+".json")
            if i == num_process - 1:
                start = ranges[i]
                end = len(topics)
                p = Process(target=work, args=(start, end, topics, output_dir, i, api_id))
            else:
                start = ranges[i]
                end = ranges[i+1]
                p = Process(target=work, args=(start, end, topics, output_dir, i, api_id))
            p.start()
            processes.append(p)
            time.sleep(36)
        for p in processes:
            p.join()
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--folders", nargs = '+')
    parser.add_argument("--api_id", type=int, required=True)


    args = parser.parse_args()
    main_process(args.folders, args.api_id)

