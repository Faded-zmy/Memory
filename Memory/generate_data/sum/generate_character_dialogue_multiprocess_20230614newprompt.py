"""
generate_character_dialogue_multiprocess_20230614newprompt.py

run:
nohup python generate_characterinfo_multiprocess.py > log/gen_log_20230608.txt 2>&1 &

20230614 related之前代码错了,要补数据;interest和not interest增加遍历. by lifeng
20230614晚上 跑全部. by lifeng
"""
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

prompts_list = ['background', 'debate', 'interest', 'not_interest', 'not_related_but_can_do',
                'conversation_with_person_not_related_novel_future_past',
                'conversation_with_person_related',
                'fore', 'out_of_worldview', 'relationship', 'nonsense_chat']
# prompts_list = ['interest', 'not_interest', 'conversation_with_person_related']
# 只要back信息
prompts_types_1 = ['background', 'debate', 'not_related_but_can_do',
                   'out_of_worldview',  'nonsense_chat']
# 要人物信息
prompts_types_2 = ['conversation_with_person_related', 'relationship']

# 遍历兴趣爱好、尊敬的人
prompts_types_3 = ['interest', 'not_interest']

# 超出世界的人物类型
prompt_not_related_types = ['future', 'past', 'novel', 'high latitude planet']

character_relationship_list = ['parents', 'marriage', 'children', 'colleague', 'classmate', 'friends', 'pet',
                               'dislikes']

# 数量配置
generate_nums ={'background':5,
                'debate':5,
                'interest':5,
                'not_interest':5,
                'nonsense_chat':5,
                'not_related_but_can_do':5,
                'conversation_with_person_not_related_novel_future_past':5,
                'out_of_worldview':5,
                'conversation_with_person_related':5,
                'relationship':20,
                'fore':40
}


def load_prompts(prompts_path):
    prompts_dict = {}
    if not os.path.exists(prompts_path):
        return prompts_dict
    for prompt_type in prompts_list:
        if os.path.exists(prompts_path + 'prompt_' + prompt_type + '.txt'):
            with open(prompts_path + 'prompt_' + prompt_type + '.txt', 'r') as f:
                prompts_dict[prompt_type] = f.read()
        else:
            print("{} not exists! Continue.".format(prompts_path + 'prompt_' + prompt_type + '.txt'))
    return prompts_dict


def generate_relationship_prompt(character_origin, prompts_dict, prompt_type,  character_name, info, relationship_type):
    prompt_template = prompts_dict[prompt_type]
    prompts_res = []
    relationship_info = info.split('\n')
    start_index = 0
    for i in range(len(relationship_info)):
        if relationship_info[i].split(':')[0].lower() == 'name':
            start_index = i
            break
    relationship_info = '\n'.join(relationship_info[start_index:]).split('--separation_line--')
    for i in range(len(relationship_info)):
        temp_relationship_info = relationship_info[i]
        split_info = temp_relationship_info.split('\n')
        for temp_info in split_info:
            if 'none' in temp_info.lower():
                return prompts_res
            if temp_info.split(':')[0].lower() == 'name':
                relationship_name = temp_info.split(':')[1].strip()
            if temp_info.split(':')[0].lower() == 'relationship' and relationship_type != 'pet':
                relationship_type = temp_info.split(':')[1].strip()
            # if temp_info.split(':')[0].lower() == 'the story between the two':
            #     relationship_story = temp_info.split(':')[1].strip()

        temp_prompt = prompt_template.format(character_origin['background'], relationship_type, temp_relationship_info,
                                             character_name, relationship_name)
        prompts_res.append(temp_prompt)
    return prompts_res


def get_interest_not_interest_prompt(character_origin, prompts_dict, prompt_type):
    character_info = character_origin['background']
    prompt_template = prompts_dict[prompt_type]
    prompt_res_list = []
    if prompt_type == 'interest':
        topics_of_interest = character_info.lower().split('topics of interest: ')[1].split('\n')[0].split(',')
        people_they_admire = character_info.lower().split('people they admire: ')[1].split('\n')[0].split(',')
        for interest in topics_of_interest:
            prompt = prompt_template.format(character_info, interest)
            prompt_res_list.append(prompt)
        for people in people_they_admire:
            prompt = prompt_template.format(character_info, people)
            prompt_res_list.append(prompt)
    elif prompt_type == 'not_interest':
        topics_of_disinterest = character_info.lower().split('topics of disinterest: ')[1].split('\n')[0].split(',')
        people_they_dislike = character_info.lower().split('people they dislike: ')[1].split('\n')[0].split(',')
        for interest in topics_of_disinterest:
            prompt = prompt_template.format(character_info, interest)
            prompt_res_list.append(prompt)
        for people in people_they_dislike:
            prompt = prompt_template.format(character_info, people)
            prompt_res_list.append(prompt)
    return prompt_res_list


def generate_prompt_reply(character_origin, character_name, prompt_type, prompt_dict, data_length, split_id, log, output_dir):
    if prompt_type in prompts_types_1:
        my_data = []
        if os.path.exists(os.path.join(output_dir, character_name.replace(" ", "_") + "_" + prompt_type + "_" + str(split_id) + ".json")):
            my_data = utils.jload(
                os.path.join(output_dir, character_name.replace(" ", "_") + "_" + prompt_type + "_" + str(split_id) + ".json"))
            log.info(f"Loaded {len(my_data)} back-generated instructions")
            print(f"Loaded {len(my_data)} back-generated instructions")
        request_idx = len(my_data)

        while len(my_data) < data_length:
            prompt = prompt_dict[prompt_type].format(character_origin['background'])
            log.info("Prompt type: {}. Prompt is: \n {}".format(prompt_type, prompt))
            result = utils.openai_chatcompletion2(prompt)
            log.info("output result: {}".format(result))
            print("\noutput result:\n", result)
            my_data.append({prompt_type: result})
            utils.jdump(my_data,
                        os.path.join(output_dir, character_name.replace(" ", "_") + "_" + prompt_type + "_" + str(split_id) + ".json"))
            request_idx += 1
            log.info("{} has done!".format(request_idx))
            print(request_idx, "has done!")
    elif prompt_type in prompts_types_2:
        my_data = []
        if os.path.exists(os.path.join(output_dir, character_name.replace(" ", "_") + "_" + prompt_type + "_" + str(split_id) + ".json")):
            my_data = utils.jload(
                os.path.join(output_dir, character_name.replace(" ", "_") + "_" + prompt_type + "_" + str(split_id) + ".json"))
            log.info(f"Loaded {len(my_data)} back-generated instructions")
            print(f"Loaded {len(my_data)} back-generated instructions")
        request_idx = len(my_data)
        while len(my_data) < data_length:
            for relationship_type in character_relationship_list:
                relationship_info = character_origin[relationship_type]
                if relationship_type == 'pet' and prompt_type == 'conversation_with_person_related':
                    continue
                prompts_relationship = generate_relationship_prompt(character_origin, prompt_dict, prompt_type, character_name, relationship_info, relationship_type)
                for prompt in prompts_relationship:
                    log.info("Prompt type: {}. Prompt is: \n {}".format(prompt_type, prompt))
                    result = utils.openai_chatcompletion2(prompt)
                    log.info("output result: {}".format(result))
                    print("\noutput result:\n", result)
                    my_data.append({prompt_type: result})
                    utils.jdump(my_data,
                                os.path.join(output_dir, character_name.replace(" ", "_") + "_" + prompt_type + "_" + str(split_id) + ".json"))
                    request_idx += 1
                    log.info("{} has done!".format(request_idx))
                    print(request_idx, "has done!")
    elif prompt_type in prompts_types_3:
        my_data = []
        if os.path.exists(os.path.join(output_dir, character_name.replace(" ", "_") + "_" + prompt_type + "_" + str(
                split_id) + ".json")):
            my_data = utils.jload(
                os.path.join(output_dir,
                             character_name.replace(" ", "_") + "_" + prompt_type + "_" + str(split_id) + ".json"))
            log.info(f"Loaded {len(my_data)} back-generated instructions")
            print(f"Loaded {len(my_data)} back-generated instructions")
        request_idx = len(my_data)
        prompts_interest_not_interest = get_interest_not_interest_prompt(character_origin, prompt_dict, prompt_type)
        for prompt in prompts_interest_not_interest:
            log.info("Prompt type: {}. Prompt is: \n {}".format(prompt_type, prompt))
            result = utils.openai_chatcompletion2(prompt)
            log.info("output result: {}".format(result))
            print("\noutput result:\n", result)
            my_data.append({prompt_type: result})
            utils.jdump(my_data,
                        os.path.join(output_dir, character_name.replace(" ", "_") + "_" + prompt_type + "_" + str(
                            split_id) + ".json"))
            request_idx += 1
            log.info("{} has done!".format(request_idx))
            print(request_idx, "has done!")
    elif prompt_type == 'fore':
        delimiters = ['\n', '\r\n']
        detailed_experiences = re.split('|'.join(map(re.escape, delimiters)), character_origin["detailed_experiences"])
        my_data = []
        if os.path.exists(os.path.join(output_dir, character_name.replace(" ", "_") + "_" + prompt_type + "_" + str(
                split_id) + ".json")):
            my_data = utils.jload(
                os.path.join(output_dir,
                             character_name.replace(" ", "_") + "_" + prompt_type + "_" + str(split_id) + ".json"))
            log.info(f"Loaded {len(my_data)} back-generated instructions")
            print(f"Loaded {len(my_data)} back-generated instructions")
        request_idx = len(my_data)

        while len(my_data) < data_length:
            prompt = prompt_dict[prompt_type].format(character_name, detailed_experiences[random.randint(0, len(detailed_experiences) - 1)])
            log.info("Prompt type: {}. Prompt is: \n {}".format(prompt_type, prompt))
            result = utils.openai_chatcompletion2(prompt)
            log.info("output result: {}".format(result))
            print("\noutput result:\n", result)
            my_data.append({prompt_type: result})
            utils.jdump(my_data,
                        os.path.join(output_dir, character_name.replace(" ", "_") + "_" + prompt_type + "_" + str(
                            split_id) + ".json"))
            request_idx += 1
            log.info("{} has done!".format(request_idx))
            print(request_idx, "has done!")
    elif prompt_type == 'conversation_with_person_not_related_novel_future_past':
        my_data = []
        if os.path.exists(os.path.join(output_dir, character_name.replace(" ", "_") + "_" + prompt_type + "_" + str(
                split_id) + ".json")):
            my_data = utils.jload(
                os.path.join(output_dir,
                             character_name.replace(" ", "_") + "_" + prompt_type + "_" + str(split_id) + ".json"))
            log.info(f"Loaded {len(my_data)} back-generated instructions")
            print(f"Loaded {len(my_data)} back-generated instructions")
        request_idx = len(my_data)
        while len(my_data) < data_length:
            for prompt_not_related_type in prompt_not_related_types:
                prompt = prompt_dict[prompt_type].format(character_origin['background'], prompt_not_related_type)
                log.info("Prompt type: {}. Prompt is: \n {}".format(prompt_type, prompt))
                result = utils.openai_chatcompletion2(prompt)
                log.info("output result: {}".format(result))
                print("\noutput result:\n", result)
                my_data.append({prompt_type: result})
                utils.jdump(my_data,
                            os.path.join(output_dir,
                                         character_name.replace(" ", "_") + "_" + prompt_type + "_" + str(
                                             split_id) + ".json"))
                request_idx += 1
                log.info("{} has done!".format(request_idx))
                print(request_idx, "has done!")



def find_word_in_string(w, s):
    return re.compile(r"\b({0})\b".format(w), flags=re.IGNORECASE).search(s)


def generate_instruction_following_data(
        output_dir="./characters/characters_dialogue_20230608",
        num_instructions_to_generate=30,
        model_name="text-davinci-003",
        num_prompt_instructions=1,
        request_batch_size=1,
        temperature=1.0,
        top_p=1.0,
        num_cpus=16,
        split_num=10,
        character_name="",
        character_path="/home/ec2-user/lifeng/DataGeneration/characters/real/",
        prompts_path="./prompts/prompts_20230613/",
        split_id=0
):
    log = logging.getLogger('feng.li')
    log.setLevel('INFO')
    date = str(datetime.now().date())
    if not os.path.exists('./log/' + date):
        os.mkdir('./log/' + date)
    file_handler = logging.FileHandler('./log/' + date + '/generate_characters_' + str(split_id) + '.log',
                                       encoding='utf-8')
    file_handler.setLevel('INFO')
    fmt_str = '%(asctime)s %(thread)d %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'
    formatter = logging.Formatter(fmt_str)
    file_handler.setFormatter(formatter)
    log.addHandler(file_handler)

    # pre prompts
    prompts_dict = load_prompts(prompts_path)
    # load the LM-generated instructions
    character_origin = json.load(open(character_path + character_name))
    character_name = character_name.split(".json")[0]

    for prompt_type in prompts_dict:
        data_length = generate_nums[prompt_type]
        try:
            generate_prompt_reply(character_origin, character_name, prompt_type, prompts_dict, data_length, split_id, log, output_dir)
        except Exception as e:
            log.error("Character_name: {}, Prompt_type: {}, Error: {}".format(character_name, prompt_type, e))
            print("Character_name: {}, Prompt_type: {}, Error: {}".format(character_name, prompt_type, e))


def work(character_name, character_path, output_dir, split_id):
    generate_instruction_following_data(character_name=character_name, character_path=character_path,
                                        output_dir=output_dir, split_id=split_id)


if __name__ == "__main__":
    prompts_path = './prompts/'

    name_list = pd.read_csv('characters/real_character_list.csv')['Name'].to_list()
    debug_mode = False
    num_processes = 15

    for i in range(len(name_list)):
        if name_list[i] != 'Monkey King (Sun Wukong) - "Journey to the West"':
            continue
        character_name = name_list[i].split(' -')[0].replace("\"", "") + '.json'
        character_path = "./characters/characters_info_20230621/"
        output_dir = "./characters/characters_dialogue_20230621/"
        if debug_mode:
            # 数量配置
            generate_nums ={'background':1,
                            'debate':1,
                            'interest':1,
                            'not_interest':1,
                            'nonsense_chat':1,
                            'not_related_but_can_do':1,
                            'conversation_with_person_not_related_novel_future_past':1,
                            'out_of_worldview':1,
                            'conversation_with_person_related':1,
                            'relationship':1,
                            'fore':1
            }
            work(character_name, character_path, output_dir, 0)
        else:
            processes = []
            for j in range(num_processes):
                # 这里设置您的character_name, character_path 和 split_id 参数
                p = Process(target=work, args=(character_name, character_path, output_dir, j))
                p.start()
                processes.append(p)

            # 等待所有进程完成
            for p in processes:
                p.join()
            print("{}'s dialogue is done.".format(character_name.split('.')[0]))

