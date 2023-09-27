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


def load_prompt(topic_ls):
    topics = ''
    for i,t in enumerate(topic_ls):
        try:
            topics += str(i+1)+'. '+t['topic']+'\n'
        except Exception as e:
            print(e)
            print(t)
    prompt = f"""
    The following are some topics that two people talked about in a day, please merge related topics into one topic and give the merged topic.

    Requirement:
        1. The merged topics are not related to each other

    The topics are:
    [Start of the topics]
    {topics}
    [End of the topics]

    The output format is as follows:
    Merged topic1: xxx
        Original topic: 
            2. xxx
            25. xxx
            ......
    Merged topic2: xxx
    ......
    """
    return prompt

def PostProcess(result):
    result = result.strip().replace('Main_Topic: ', '')
    return result


def generate_prompt_reply(start, end, combinations, output_dir, i, api_id):
    # topic_ls = combinations[start:end]
    # if not os.path.exists(os.path.join(output_dir, f'{i}.json')):
    my_data = []
    # idx = 0
    right_end,left_end = start,start
    org_weights = [round(0.1+0.2*w,2) for w in range(20)]+[round(4.1-0.2*w,2) for w in range(21)]
    weights=[round(ow/sum(org_weights),4) for ow in org_weights]
    while right_end<end:
    # for org_conv in org_convs:
        left_end = right_end
        num = random.choices( [number for number in range(10,51)],weights=weights) [0]
        right_end = left_end + num
        topic_ls = combinations[left_end:min(end,right_end)]
        prompt = load_prompt(topic_ls)
        # print('-'*60)
        # print("prompt:", prompt)
        # print('-'*60)
        result = openai_chatcompletion(prompt,api_id)
        # log.info("output result: {}".format(result))
        # print('-'*60)
        # print("\noutput result:\n", result)
        # print('-'*60)
        my_data.append({'topic_ls':topic_ls, 'result':result})
        # idx+=1
    # if my_data != {'A_info':[], 'B_info':[], 'Dis':[], 'org_conv':[]}:
    utils.jdump(my_data,
                os.path.join(output_dir, f'{i}.json'))
    # request_idx += 1
    # log.info("{} has done!".format(len(my_data)))
    print(len(my_data), "has done!")
    




# def generate_instruction_following_data(
#         json_path,
#         output_dir="./characters/characters_dialogue_20230608",
#         split_id=0,
#         api_id=0
# ):
#     log = logging.getLogger('mengying.zhou')
#     log.setLevel('INFO')
#     date = str(datetime.now().date())
#     if not os.path.exists('/ai_jfs/mengying/data/sum/log/log1/' + date):
#         os.mkdir('/ai_jfs/mengying/data/sum/log/log1/' + date)
#     file_handler = logging.FileHandler('/ai_jfs/mengying/data/sum/log/log1/' + date + '/generate_characters_' + str(split_id) + '.log', encoding='utf-8')
#     file_handler.setLevel('INFO')
#     fmt_str = '%(asctime)s %(thread)d %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'
#     formatter = logging.Formatter(fmt_str)
#     file_handler.setFormatter(formatter)
#     log.addHandler(file_handler)

#     # pre prompts
    
#     try:
#         generate_prompt_reply(json_path, output_dir, split_id, log, api_id)
#     except Exception as e:
#         log.error("Json_path: {}, Error: {}".format(json_path, e))
#         # print("Json_path: {}, Error: {}".format(json_path, e))


# def work(start, end, combinations, output_dir, split_id, api_id):
#             generate_instruction_following_data(json_path=json_path,
#                                                 output_dir=output_dir, split_id=split_id,
#                                                 api_id=api_id)


def main_process(data_path, api_id):
    combinations = json.load(open(data_path, 'r'))
    random.shuffle(combinations) ## shuffle the data
    num_process = 15
    division = len(combinations) // num_process
    ranges = [i*division for i in range(num_process)]
    processes = []
    output_dir = '/ai_jfs/mengying/data/sum/topic_group/shuffle/'
    for i in range(num_process):
        if i == num_process - 1:
            start = ranges[i]
            end = len(combinations)
            p = Process(target=generate_prompt_reply, args=(start, end, combinations, output_dir, i, api_id))
        else:
            start = ranges[i]
            end = ranges[i+1]
            p = Process(target=generate_prompt_reply, args=(start, end, combinations, output_dir, i, api_id))
        p.start()
        processes.append(p)
        time.sleep(36)
    for p in processes:
        p.join()
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--data_path", nargs = '+')
    parser.add_argument("--data_path", type = str)
    parser.add_argument("--api_id", type=int, required=True)


    args = parser.parse_args()
    main_process(args.data_path, args.api_id)

