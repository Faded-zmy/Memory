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


def openai_chatcompletion(prompt, api_id):
    api_key = [
    # "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",
    # "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",
    "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",
    # "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",
    ]
    openai.api_key = api_key[api_id]
    cn = 0
    enc = tiktoken.get_encoding("cl100k_base")
    print(f"Tokens Of Prompt in {api_id}: {len(enc.encode(prompt))}!")
    if len(enc.encode(prompt)) > 7000:
        return None
    # max_token = 15000 - len(enc.encode(prompt))

    while True:
        output = ""
        try:
            response = openai.ChatCompletion.create(
                # model="gpt-3.5-turbo-16k",
                model="gpt-4",
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


def load_prompt(conv):
    prompt = """
    The following is a conversation between User_A and User_B. Please help to extract information and summarize opinion on both sides from the following conversation.
    Requirement:
    1. Divide the information into two parts, a description of people and opinions on a topic.
    2. Summarize the conversation into one topic and the opinion of both sides. Give both sides's way of talking and did he achieve a achievement such as convincing someone, getting a message when discuss the topic using a concise sentence or some words.
    3. For the description of people, construct an information card of both sides.
    4. "Todo" is what people is going to do, "Todo_Time" is corresponding time.
    5. If a certain key's information is not mentioned, fill in it with "None".

    The structure of the information card is as follows:
    {"basic_information": {"Name": xxx, "Gender": xxx, "Date_of_Birth": xxx, "Place_of_Birth": xxx, "Race": xxx, "Nationality": xxx}, "background": {"Educational_Background": xxx, "Occupation": xxx, "Position": xxx, "Achievement": xxx}, "others": {"Personality": xxx, "Hobbies": xxx, "Good_at": xxx, "Not_good_at": xxx, "Topics_of_interest": xxx, "Topics_of_disinterest": xxx, "People_They_Admire": xxx, "People_They_Dislike": xxx, "Todo": xxx, "Todo_Time": xxx}}}

    The output format is as follows:
    User_A's information card: xxx

    User_B's information card: xxx

    Discussions:
    {"Topic": xxx,
    "Summary": xxx,
    "User_A's_opinion": xxx,
    "User_A's_way_of_talking": xxx,
    "User_A's_achievement":xxx,
    "User_B's_opinion": xxx,
    "User_B's_way_of_talking": xxx,
    "User_B's_achievement":xxx}

    The conversation is as follows:\n
    """
    prompt = prompt+conv
    return prompt

def PreProcess(org_conv):
    final_conv = re.sub(r'\n.*Inner thoughts:.*', "", org_conv, flags=re.I)
    final_conv = re.sub(r'\n.*actions/expressions:.*', "", final_conv, flags=re.I)
    final_conv = re.sub('\n.*answer:', "", final_conv, flags=re.I)
    final_conv = re.sub('Person', "User_A", final_conv, flags=re.I)
    final_conv = re.sub('Character', "User_B", final_conv, flags=re.I)
    return final_conv

def PostProcess(response):
    #转成dic
    splitted_data = re.split(f"(User_A's information card|User_B's.*information card|Discussions):", response, flags=re.I)
    A_info_card = splitted_data[2].strip().replace(': None',': "None"')
    # print("A_info_card", A_info_card)
    B_info_card = splitted_data[4].strip().replace(': None',': "None"')
    Discussions = splitted_data[6].strip().replace(': None',': "None"')
    A_json = json.loads(A_info_card)
    B_json = json.loads(B_info_card)
    discussion_json = json.loads(Discussions)
    
    #压缩key
    key_map = {
        "basic_information": "bi",
        "background": "bg",
        "others": "o",
        "name": "N",
        "gender": "Gd",
        "date_of_birth": "DB",
        "place_of_birth": "PB",
        "race": "R",
        "nationality": "Nat",
        "educational_background": "EB",
        "occupation": "Op",
        "position": "Pos",
        "achievement": "A",
        "personality": "Per",
        "hobbies": "H",
        "good_at": "G",
        "not_good_at": "Ng",
        "topics_of_interest": "Toi",
        "topics_of_disinterest": "Tod",
        "people_they_admire": "PTA",
        "people_they_dislike": "PTD",
        "todo": "Td",
        "todo_time": "TT",
        "topic": "T",
        "user_a's_opinion": "Ao",
        "user_b's_opinion": "Bo",
        "user_a's_way_of_talking": "Aw",
        "user_b's_way_of_talking": "Bbao bw",
        "summary": "sum",
        "user_a's_achievement": "Aa",
        "user_b's_achievement": "Ba"
    }
    A_final = {}
    B_final = {}
    Discussions_final = {}
    for key0 in A_json.keys():
        A_final[key_map[key0.lower().replace(' ','_')]] = {}
        B_final[key_map[key0.lower().replace(' ','_')]] = {}
    for key0 in A_json.keys():
        for key1 in A_json[key0].keys():
            A_final[key_map[key0.lower().replace(' ','_')]][key_map[key1.lower().replace(' ','_')]] = A_json[key0][key1]
            B_final[key_map[key0.lower().replace(' ','_')]][key_map[key1.lower().replace(' ','_')]] = B_json[key0][key1]
    for key in discussion_json.keys():
        Discussions_final[key_map[key.lower().replace(' ','_')]] = discussion_json[key]
    return A_final, B_final, Discussions_final




def generate_prompt_reply(json_path, output_dir, split_id, log):
    my_data = {'A_info':[], 'B_info':[], 'Dis':[], 'org_conv':[]}
    org_convs = json.load(open(json_path, 'r'))
    request_idx = len(my_data)

    for org_conv in org_convs:
        final_conv = PreProcess(list(org_conv.values())[0])
        if final_conv == "":
            continue
        # print("\noutput final_conv:\n", final_conv)
        prompt = load_prompt(final_conv)
        log.info("Prompt is: \n {}".format(prompt))
        result = openai_chatcompletion(prompt,0)
        log.info("output result: {}".format(result))
        # print("\noutput result:\n", result)
        try:
            A_info, B_info, Dis = PostProcess(result)
            log.info("A_info result: {}".format(A_info))
            # print("\nA_info result:\n", A_info)
            my_data['A_info'].append(A_info)
            my_data['B_info'].append(B_info)
            my_data['Dis'].append(Dis)
            my_data['org_conv'].append(final_conv)
        except Exception:
            print("Format Mistake !!!")
    if my_data != {'A_info':[], 'B_info':[], 'Dis':[], 'org_conv':[]}:
        utils.jdump(my_data,
                    os.path.join(output_dir, json_path.split('/')[-1]))
        # request_idx += 1
        log.info("{} has done!".format(len(my_data)))
        print(len(my_data), "has done!")
    




def generate_instruction_following_data(
        json_path,
        output_dir="./characters/characters_dialogue_20230608",
        split_id=0
):
    log = logging.getLogger('mengying.zhou')
    log.setLevel('INFO')
    date = str(datetime.now().date())
    if not os.path.exists('/ai_jfs/mengying/data/sum/log/log2/' + date):
        os.mkdir('/ai_jfs/mengying/data/sum/log/log2/' + date)
    file_handler = logging.FileHandler('/ai_jfs/mengying/data/sum/log/log2/' + date + '/generate_characters_' + str(split_id) + '.log', encoding='utf-8')
    file_handler.setLevel('INFO')
    fmt_str = '%(asctime)s %(thread)d %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'
    formatter = logging.Formatter(fmt_str)
    file_handler.setFormatter(formatter)
    log.addHandler(file_handler)

    # pre prompts
    
    try:
        generate_prompt_reply(json_path, output_dir, split_id, log)
    except Exception as e:
        log.error("Json_path: {}, Error: {}".format(json_path, e))
        # print("Json_path: {}, Error: {}".format(json_path, e))


def work(start, end, combinations, output_dir, split_id):
    
    for i in tqdm(range(start, end), total=end-start):
        json_path = combinations[i]
        print(json_path)
        if json_path.split('.')[-1] == 'json' and not os.path.exists(os.path.join(output_dir, json_path.split('/')[-1])):
            print(json_path)
            generate_instruction_following_data(json_path=json_path,
                                                output_dir=output_dir, split_id=split_id)


def main():
    # with open('roomchat_person_selected_news.json', 'r', encoding='utf-8') as f:
    #     combinations = utils.jload(f)
    # print(combinations)
    folders = ['/ai_jfs/lifeng/data/character/characters_dialogue/characters_dialogue_20230608_gpu1']
    for root_path in folders:
        # root_path = '/ai_efs/lifeng/data/character/characters_dialogue/characters_dialogue_20230705_gpu2/'
        combinations = [root_path+'/'+file for file in os.listdir(root_path) if file.endswith('.json')]
        num_process = 7
        division = len(combinations) // num_process
        ranges = [i*division for i in range(num_process)]
        processes = []
        output_dir = '/ai_jfs/mengying/data/sum/'+root_path.split('/')[-1]+'/'
        for i in range(num_process):
            # save_path = os.path.join(output_dir, "regen_roomchat_specific_news_"+str(i)+".json")
            if i == num_process - 1:
                start = ranges[i]
                end = len(combinations)
                p = Process(target=work, args=(start, end, combinations, output_dir, i))
            else:
                start = ranges[i]
                end = ranges[i+1]
                p = Process(target=work, args=(start, end, combinations, output_dir, i))
            p.start()
            processes.append(p)
            time.sleep(36)
        for p in processes:
            p.join()
 
if __name__ == '__main__':
    main()

