"""Extension that allows us to fetch and store memories from/to LTM."""

import json
import os
import time
import pathlib
import pprint
from typing import List, Tuple

import gradio as gr

import modules.shared as shared
# from modules.chat import generate_chat_prompt
from modules.html_generator import fix_newlines

from extensions.long_term_memory.core.memory_database_new import LtmDatabase
from extensions.long_term_memory.utils.chat_parsing import clean_character_message
from extensions.long_term_memory.utils.timestamp_parsing import (
    get_time_difference_message,
)
from extensions.long_term_memory.constants import (
    # _CONVERSATION_ROUND_TO_SUM,
    _OVERLAPPED_ROUNDS,
    _CONVERSATION_WORDS_TO_SUM,
    _MEM_ROOT
)
from extensions.long_term_memory.samsum import samsum


# === Internal constants (don't change these without good reason) ===
_CONFIG_PATH = "extensions/long_term_memory/ltm_config_new.json"
_MIN_ROWS_TILL_RESPONSE = 5
_LAST_BOT_MESSAGE_INDEX = -3
conv_round,sum_round,last_date = 0,0,''
convs = ''
c=time.localtime()
d=time.strftime("%Y-%m-%d",c)
# if os.path.exists(TMP_CONV):
#     tmp = json.load(open(TMP_CONV,'r'))
#     if tmp!= {} and tmp['date'] == d:
#         sum_round = tmp['sum_round']
#         last_date = d


    
   

_LTM_STATS_TEMPLATE = """{num_memories_seen_by_bot} memories are loaded in the bot
{num_memories_in_ram} memories are loaded in RAM
{num_memories_on_disk} memories are saved to disk"""
with open(_CONFIG_PATH, "rt") as handle:
    _CONFIG = json.load(handle)


# === Module-level variables ===
debug_texts = {
    "current_memory_text": "(None)",
    "num_memories_loaded": 0,
    "current_context_block": "(None)",
}
memory_database = LtmDatabase(
    pathlib.Path(_MEM_ROOT),
    num_memories_to_fetch=_CONFIG["ltm_reads"]["num_memories_to_fetch"],
)
# This bias string is currently unused, feel free to try using it
params = {
    "activate": False,
    "bias string": " *I got a new memory! I'll try bringing it up in conversation!*",
}


# === Display important notes to the user ===
print()
print("-----------------------------------------")
print("IMPORTANT LONG TERM MEMORY NOTES TO USER:")
print("-----------------------------------------")
print(
    "Please remember that LTM-stored memories will only be visible to "
    "the bot during your NEXT session. This prevents the loaded memory "
    "from being flooded with messages from the current conversation which "
    "would defeat the original purpose of this module. This can be overridden "
    "by pressing 'Force reload memories'"
)
print("----------")
print("LTM CONFIG")
print("----------")
print("change these values in ltm_config.json")
pprint.pprint(_CONFIG)
print("----------")
print("-----------------------------------------")


def _get_current_memory_text() -> str:
    return debug_texts["current_memory_text"]


def _get_num_memories_loaded() -> int:
    return debug_texts["num_memories_loaded"]


# def _get_current_ltm_stats() -> str:
#     num_memories_in_ram = memory_database.message_embeddings.shape[0] \
#             if memory_database.message_embeddings is not None else "None"
#     num_memories_on_disk = memory_database.disk_embeddings.shape[0] \
#             if memory_database.disk_embeddings is not None else "None"

#     ltm_stats = {
#         "num_memories_seen_by_bot": _get_num_memories_loaded(),
#         "num_memories_in_ram": num_memories_in_ram,
#         "num_memories_on_disk": num_memories_on_disk,
#     }
#     ltm_stats_str = _LTM_STATS_TEMPLATE.format(**ltm_stats)
#     return ltm_stats_str


def _get_current_context_block() -> str:
    return debug_texts["current_context_block"]


def _build_augmented_context(memory_context: str, original_context: str) -> str:
    injection_location = _CONFIG["ltm_context"]["injection_location"]
    if injection_location == "BEFORE_NORMAL_CONTEXT":
        augmented_context = f"{memory_context.strip()}\n{original_context.strip()}"
    elif injection_location == "AFTER_NORMAL_CONTEXT_BUT_BEFORE_MESSAGES":
        if "<START>" not in original_context:
            raise ValueError(
                "Cannot use AFTER_NORMAL_CONTEXT_BUT_BEFORE_MESSAGES, "
                "<START> token not found in context. Please make sure you're "
                "using a proper character json and that you're NOT using the "
                "generic 'Assistant' sample character"
            )

        split_index = original_context.index("<START>")
        augmented_context = original_context[:split_index] + \
                memory_context.strip() + "\n" + original_context[split_index:]
    else:
        raise ValueError(f"Invalid injection_location: {injection_location}")

    return augmented_context


# === Hooks to oobaboogs UI ===
def bot_prefix_modifier(string):
    """
    This function is only applied in chat mode. It modifies
    the prefix text for the Bot and can be used to bias its
    behavior.
    """
    if params["activate"]:
        bias_string = params["bias string"].strip()
        return f"{string} {bias_string} "
    return string


def ui():
    """Adds the LTM-specific settings."""
    with gr.Accordion("Long Term Memory settings", open=True):
        with gr.Row():
            update = gr.Button("Force reload memories")
    with gr.Accordion(
        "Long Term Memory debug status (must manually refresh)", open=True
    ):
        with gr.Row():
            current_memory = gr.Textbox(
                value=_get_current_memory_text(),
                label="Current memory loaded by bot",
            )
            # current_ltm_stats = gr.Textbox(
            #     value=_get_current_ltm_stats(),
            #     label="LTM statistics",
            # )
        with gr.Row():
            current_context_block = gr.Textbox(
                value=_get_current_context_block(),
                label="Current FIXED context block (ONLY includes example convos)"
            )
        with gr.Row():
            refresh_debug = gr.Button("Refresh")
    with gr.Accordion("Long Term Memory DANGER ZONE (don't do this immediately after switching chars, write a msg first)", open=False):
        with gr.Row():
            destroy = gr.Button("Destroy all memories", variant="stop")
            destroy_confirm = gr.Button(
                "THIS IS IRREVERSIBLE, ARE YOU SURE?", variant="stop", visible=False
            )
            destroy_cancel = gr.Button("Do Not Delete", visible=False)
            destroy_elems = [destroy_confirm, destroy, destroy_cancel]

    # Update memories
    update.click(memory_database.reload_embeddings_from_disk, [], [])

    # Update debug info
    refresh_debug.click(fn=_get_current_memory_text, outputs=[current_memory])
    # refresh_debug.click(fn=_get_current_ltm_stats, outputs=[current_ltm_stats])
    refresh_debug.click(fn=_get_current_context_block, outputs=[current_context_block])

    # Clear memory with confirmation
    destroy.click(
        lambda: [gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)],
        None,
        destroy_elems,
    )
    destroy_confirm.click(
        lambda: [gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)],
        None,
        destroy_elems,
    )
    destroy_confirm.click(memory_database.destroy_all_memories, [], [])
    destroy_cancel.click(
        lambda: [gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)],
        None,
        destroy_elems,
    )

#大头
# def _build_memory_context(fetched_memories: List[Tuple[str, float]], name1: str, name2: str):
#     memory_length_cutoff = _CONFIG["ltm_reads"]["memory_length_cutoff_in_chars"]

#     # Build all the individual memory strings
#     memory_strs = []
#     distance_scores = []
#     debug_texts["current_memory_text"] = "(None)"
#     debug_texts["num_memories_loaded"] = 0
#     for (fetched_memory, distance_score) in fetched_memories:
#         if fetched_memory and distance_score < _CONFIG["ltm_reads"]["max_cosine_distance"]:
#             time_difference = get_time_difference_message(fetched_memory["timestamp"])
#             memory_str = _CONFIG["ltm_context"]["memory_template"].format(
#                 time_difference=time_difference,
#                 memory_message=fetched_memory["message"][:memory_length_cutoff],
#             )
#             memory_strs.append(memory_str)
#             distance_scores.append(distance_score)

#     # No memories fetched, we'll have no memory_context
#     if not memory_strs:
#         return None

#     # Now inject all memory strings into the wider memory context
#     joined_memory_strs = "\n".join(memory_strs)
#     memory_context = _CONFIG["ltm_context"]["memory_context_template"].format(
#         name1=name1,
#         name2=name2,
#         all_memories=joined_memory_strs,
#     )

#     # Report debugging info to user
#     print("------------------------------")
#     print("NEW MEMORIES LOADED IN CHATBOT")
#     pprint.pprint(joined_memory_strs)
#     debug_texts["current_memory_text"] = joined_memory_strs
#     debug_texts["num_memories_loaded"] = len(memory_strs)
#     print("scores (in order)", distance_scores)
#     print("------------------------------")
#     return memory_context

def _build_memory_context(fetched_memories: List[Tuple[str, float]]):
    # print('fetched_memories',fetched_memories)
    memory_org_conv = ''
    memory_context = ''
    joined_memory_strs = ''
    for i,(fetched_memory, distance_score) in enumerate(fetched_memories):
        if not os.path.exists(ORIGINAL_CONV):
            os.system('mkdir {}'.format(ORIGINAL_CONV))
        conv_path = ORIGINAL_CONV+'/'+fetched_memory['timestamp']+'-sum_round'+str(fetched_memory['sum_round']-fetched_memory['sum_round']%100)+'.json'
        # print('conv_path',conv_path)
        memory_org_conv += json.load(open(conv_path,'r'))['sum_round{}'.format(fetched_memory['sum_round'])]
        memory_context += fetched_memory['message']
        joined_memory_strs += 'RELEVANT_SUM'+str(i)+fetched_memory['message']+'\n'
    debug_texts["current_memory_text"] = joined_memory_strs
    # memory_org_conv = ''.join(f[int(memory_context.split('round')[1].split(' ')[0])-_CONVERSATION_ROUND_TO_SUM:int(memory_context.split('round')[1].split(' ')[0])])
    augmented_context = "Based on the provided context summary and the original conversation, provide the answer to the question.\nContextual summary:"+memory_context+'\n'+'Original conversation:\n'+memory_org_conv+'\nQuestion:\n'
 
    return augmented_context

# Thanks to @oobabooga for providing the fixes for:
# https://github.com/wawawario2/long_term_memory/issues/12
# https://github.com/wawawario2/long_term_memory/issues/14
# https://github.com/wawawario2/long_term_memory/issues/19
def custom_generate_chat_prompt(
    user_input,
    # state,
    **kwargs,
):
    global convs,sum_round,last_date,_MEM_ROOT,ORIGINAL_CONV,TMP_CONV
    # print('conv_round',conv_round)
    """Main hook that allows us to fetch and store memories from/to LTM."""
    print("=" * 60)
    
  
    if shared.settings['name1'] == '':
        raise ValueError('Please input your name!')
    else:
        _MEM_ROOT = f"./extensions/long_term_memory/user_data/bot_memories_{shared.settings['name1'].replace(' ','_')}/"
    print("MY NAME:",shared.settings['name1'])

    # if hasattr(shared, 'mem_root'):
    #     _MEM_ROOT = shared.mem_root
    #     if not os.path.exists(_MEM_ROOT):
    #         os.system('mkdir {}'.format(_MEM_ROOT))
    # else:
    #     _MEM_ROOT = "./extensions/long_term_memory/user_data/bot_memories/"
    ORIGINAL_CONV = _MEM_ROOT+'/original_conv/'
    TMP_CONV = _MEM_ROOT+'/tmp_conv.json'
    if not os.path.exists(ORIGINAL_CONV):
        os.system('mkdir {}'.format(ORIGINAL_CONV))
    # if not os.path.exists(TMP_CONV):
    #     os.system('mkdir {}'.format(TMP_CONV))

    character_name = shared.settings["name2"].strip().lower().replace(" ", "_")
    #character_name is assistant and you
    memory_database.load_character_db_if_new(character_name)
    memory_database.load_path_db_if_new(pathlib.Path(_MEM_ROOT))

    user_input = fix_newlines(user_input)

    # === Fetch the "best" memory from LTM, if there is one ===
    fetched_memories = memory_database.query(
        user_input
    )
    print("lw debug fetched_memories",fetched_memories)
    
    #这个地方是最重要的，从里面搜最重要的，明天改
    
    # === Call oobabooga's original generate_chat_prompt ===
    #context 'This is a conversation with your Assistant. It is a computer program designed to help you with various tasks such as answering questions, providing recommendations, and helping with decision making. You can ask it anything you want and it will do its best to give you accurate and relevant information.'
    augmented_context = shared.settings["context"]
    if fetched_memories != []:
        augmented_context = _build_memory_context(fetched_memories)
        # print('memory_context',augmented_context)
        # augmented_context = _build_augmented_context(memory_context, state["context"])
        debug_texts["current_context_block"] = augmented_context

    kwargs["also_return_rows"] = True
    shared.settings["context"] = augmented_context
    print("ZMY_LTM",augmented_context)
    # (prompt, prompt_rows) = generate_chat_prompt(
    #     user_input,
    #     state,
    #     **kwargs,
    # )
    prompt_rows = shared.history['visible']
    print("ZMY_PROMPT_ROWS",shared.history)
    # prompt = '\n'.join(prompt_rows) if prompt_rows!=[] else ''
    prompt = augmented_context
    


    #print("lw debug prompt_rows",prompt_rows)

    c=time.localtime()
    d=time.strftime("%Y-%m-%d",c)

    
    # if len(clean_bot_message) >= _CONFIG["ltm_writes"]["min_message_length"] and len(user_input) >= _CONFIG["ltm_writes"]["min_message_length"]:
    # print('PROMPT_ROWS',len(prompt_rows),prompt_rows)
    if len(prompt_rows)>=_MIN_ROWS_TILL_RESPONSE: #原来有的
        
        # conv_round+=1
        
        #获取对话
        bot_message = prompt_rows[_LAST_BOT_MESSAGE_INDEX]
        clean_bot_message = clean_character_message(shared.settings["name2"], bot_message)
        user_input =  prompt_rows[_LAST_BOT_MESSAGE_INDEX-1]
        clean_user_input = clean_character_message(shared.settings["name1"], user_input)
        # print('GET_CONV',"{}:{}\n{}:{}".format(state["name1"],clean_user_input,state["name2"],clean_bot_message))
        # 存入tmp中（可能会很慢）
        if os.path.exists(TMP_CONV):# and ('conversation' in json.load(open(TMP_CONV,'r')).keys()):
            tmp_conv = json.load(open(TMP_CONV,'r'))
            if 'conversation' in json.load(open(TMP_CONV,'r')).keys():
                tmp_conv['conversation'].append("{}:{}\n{}:{}".format(shared.settings["name1"],clean_user_input,shared.settings["name2"],clean_bot_message))
            else:
                tmp_conv['conversation'] = ["{}:{}\n{}:{}".format(shared.settings["name1"],clean_user_input,shared.settings["name2"],clean_bot_message)]
            # if len(tmp_conv['conversation']) == _CONVERSATION_ROUND_TO_SUM:
            print('*'*80)
            print('TOKEN NUM',sum([len(tp.split(' ')) for tp in tmp_conv['conversation']]))
            print('*'*80)
            if sum([len(tp.split(' ')) for tp in tmp_conv['conversation']]) >= _CONVERSATION_WORDS_TO_SUM:
                #确定sum_round是不是从新的一天开始计数
                if tmp_conv['last_date'] == d:
                    tmp_conv['sum_round'] += 1
                else:
                    tmp_conv['sum_round'] = 1
                    tmp_conv['last_date'] = d
                
                
                #tmp_conv满固定轮数了，提摘要存进db里
                
                convs = '\n'.join(tmp_conv['conversation'])
                sums = samsum(convs)
                name = '{}-sum_round{}'.format(d,str(tmp_conv['sum_round']))

                # print('zmy-name',name)
                memory_database.add(name,sums)
                
                # json.dump({'sum_round':tmp_conv['sum_round'],'last_date':tmp_conv['last_date']},open(TMP_CONV, 'w'))
                #分块有overlapped
                json.dump({'conversation':tmp_conv['conversation'][-_OVERLAPPED_ROUNDS:],'sum_round':tmp_conv['sum_round'],'last_date':tmp_conv['last_date']},open(TMP_CONV, 'w'))
                print("-----------------------")
                print("NEW MEMORY SAVED to LTM")
                print("-----------------------")
                print("SUM:",sums)
                print("-----------------------")

                #把原对话存进相应日期及轮数的json文件中
                org_json = ORIGINAL_CONV+"{}-sum_round{}.json".format(d,str(tmp_conv['sum_round']-tmp_conv['sum_round']%100))
                if os.path.exists(org_json):
                    org_conv = json.load(open(org_json,'r'))
                    org_conv['sum_round{}'.format(str(tmp_conv['sum_round']))] = convs
                    json.dump(org_conv,open(org_json,'w'),indent=3)
                else:
                    org_conv={'sum_round{}'.format(str(tmp_conv['sum_round'])):convs}
                    json.dump(org_conv,open(org_json,'w'),indent=3)


                    
            else:
                json.dump(tmp_conv,open(TMP_CONV,'w'),indent=3)


        else:
            tmp_conv={'last_date':d,'sum_round':0,'conversation':["{}:{}\n{}:{}".format(shared.settings["name1"],clean_user_input,shared.settings["name2"],clean_bot_message)]}
            json.dump(tmp_conv,open(TMP_CONV,'w'),indent=3)
        
        

        # convs = convs+"{}:{}\n{}:{}\n".format(state["name1"],clean_user_input,state["name2"],clean_bot_message)


        # memory_database.add(state["name2"], clean_bot_message, state["name1"], clean_user_input)
        
    # if conv_round>0 and conv_round % _CONVERSATION_ROUND_TO_SUM == 0:
        # if last_date == d:
        #     sum_round+=1
        # else:
        #     sum_round = 1
        #     last_date = d
        
        # org_json = ORIGINAL_CONV+"{}-sum_round{}.json".format(d,str(sum_round-sum_round%100))
        # if os.path.exists(org_json):
        #     org_conv = json.load(open(org_json,'r'))
        #     org_conv['sum_round{}'.format(str(sum_round))] = convs
        #     json.dump(org_conv,open(org_json,'w'),indent=3)
        # else:
        #     org_conv={'sum_round{}'.format(str(sum_round)):convs}
        #     json.dump(org_conv,open(org_json,'w'),indent=3)

        # sums = samsum(convs)
        # name = '{}-sum_round{}'.format(d,str(sum_round))

        # print('zmy-name',name)
        # memory_database.add(name,sums)
        
        # json.dump({},open(TMP_CONV, 'w'))
        # print("-----------------------")
        # print("NEW MEMORY SAVED to LTM")
        # print("-----------------------")
        # print("SUM:",sums)
        # print("-----------------------")
    #     convs = ''
        
    # print('='*60)
    # print('ZMY_LTM_PROMPT',prompt)
    # print('='*60)
    return prompt
