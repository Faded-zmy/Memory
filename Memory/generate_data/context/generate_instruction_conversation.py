"""
batch_selfinstruct_generate.py

run:
python -m generate_instruction generate_instruction_following_data \
  --output_dir ./ \
  --num_instructions_to_generate 10 \
  --model_name="text-davinci-003" \
"""
import time
import json
import os
import random
import re
import string
from functools import partial
from multiprocessing import Pool

import numpy as np
import tqdm
import utils

import fire


def encode_prompt(prompt_file):
    """Encode multiple prompt instructions into a single string."""
    prompt = open(prompt_file).read()
    return prompt


def post_process_gpt3_response(response):
    if response == "":
        return []
    raw_instructions = f"1. Setting:\n" + response
    print("response:\n", raw_instructions)
    raw_instructions = re.split("Setting:", raw_instructions)
    print("raw_instructions",raw_instructions)
    instructions = []
    for idx, inst in enumerate(raw_instructions[1:]):
    #     splitted_data = re.split(f"(Project|Business|Pain|Status|Goal):", inst)
    #     #print(idx, splitted_data)
    #     if len(splitted_data) != 11:
    #         continue
    #     else:
    #         project  = splitted_data[2].strip()
    #         business = splitted_data[4].strip()
    #         pain     = splitted_data[6].strip()
    #         status   = splitted_data[8].strip()
    #         goal     = splitted_data[10].strip().split("\n\n")[0]
    #         #print(idx, "\n", project, "\n", business, "\n", pain, "\n", status, "\n", goal)
        inst = inst.strip()
        topic = inst.split('\n')[0].split(':')[-1]
        conv   = '\n'.join(inst.split('\n')[1:])
        instructions.append({"topic": topic, "conversation": conv})
    print(instructions[-1])
    return instructions


def find_word_in_string(w, s):
    return re.compile(r"\b({0})\b".format(w), flags=re.IGNORECASE).search(s)


def generate_instruction_following_data(
    output_dir="./",
    num_instructions_to_generate=5,
    model_name="text-davinci-003",
    num_prompt_instructions=1,
    request_batch_size=1,
    temperature=1.0,
    top_p=1.0,
    num_cpus=16,
    output_file="./conversation.json",
    prompt_file="./prompt_dao_conversation.txt"
):
    #seed_tasks = [json.loads(l) for l in open(seed_tasks_path, "r")]

    os.makedirs(output_dir, exist_ok=True)
    request_idx = 0
    # load the LM-generated instructions
    machine_instruction_data = []
    if os.path.exists(os.path.join(output_dir, output_file)):
        machine_instruction_data = utils.jload(os.path.join(output_dir, output_file))
        print(f"Loaded {len(machine_instruction_data)} machine-generated instructions")

    # while len(machine_instruction_data) < num_instructions_to_generate:
    while request_idx < num_instructions_to_generate:
        request_idx += 1

        prompt = encode_prompt(prompt_file)
        print("\nprompt", prompt)
        result = utils.openai_chatcompletion2(prompt)
        print("result", result)
        new_instructions = post_process_gpt3_response(result)
        print(new_instructions)
        # for instruction_data_entry in new_instructions:
        machine_instruction_data.extend(new_instructions)
            #progress_bar.update(1)
        print("total num:", len(machine_instruction_data))
    utils.jdump(machine_instruction_data, os.path.join(output_dir, output_file))

def main(task, **kwargs):
    globals()[task](**kwargs)


if __name__ == "__main__":
    fire.Fire(main)
