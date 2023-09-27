import ast
import base64
import copy
import functools
import io
import json
import re
import torch
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageOps

import yaml
from PIL import Image

import modules.shared as shared
from modules.extensions import apply_extensions
# from modules.html_generator import chat_html_wrapper, make_thumbnail
# from modules.logging_colors import logger
# from modules.text_generation import (generate_reply, get_encoded_length,
#                                      get_max_prompt_length)
from modules.utils import replace_all
from modules.models import load_base_model
from modules.LoRA import add_lora_to_model
import requests
from modules.utils import get_available_chat_styles
import markdown

# Custom chat styles
chat_styles = {}
for k in get_available_chat_styles():
    chat_styles[k] = open(Path(f'css/chat_style-{k}.css'), 'r').read()


def replace_blockquote(m):
    return m.group().replace('\n', '\n> ').replace('\\begin{blockquote}', '').replace('\\end{blockquote}', '')


def convert_to_markdown(string):

    # Blockquote
    pattern = re.compile(r'\\begin{blockquote}(.*?)\\end{blockquote}', re.DOTALL)
    string = pattern.sub(replace_blockquote, string)

    # Code
    string = string.replace('\\begin{code}', '```')
    string = string.replace('\\end{code}', '```')
    string = re.sub(r"(.)```", r"\1\n```", string)

    result = ''
    is_code = False
    for line in string.split('\n'):
        if line.lstrip(' ').startswith('```'):
            is_code = not is_code

        result += line
        if is_code or line.startswith('|'):  # Don't add an extra \n for tables or code
            result += '\n'
        else:
            result += '\n\n'

    if is_code:
        result = result + '```'  # Unfinished code block

    string = result.strip()
    return markdown.markdown(string, extensions=['fenced_code', 'tables'])


def generate_cai_chat_html():
    reset_cache=False
    # chat_styles = {}
    # for k in get_available_chat_styles():
    #     chat_styles[k] = open(Path(f'css/chat_style-{k}.css'), 'r').read()
    # print('ZMY-style',chat_styles.keys())
    style = 'cai-chat'
    output = f'<style>{chat_styles[style]}</style><div class="chat" id="chat">'

    # We use ?name2 and ?time.time() to force the browser to reset caches
    # img_bot = f'<img src="file/cache/pfp_character.png?{name2}">' if Path("cache/pfp_character.png").exists() else ''
    img_me = f'<img src="file/cache/pfp_me.png?{time.time() if reset_cache else ""}">' if Path("cache/pfp_me.png").exists() else ''
    history = shared.history
    visible = history['visible']
    name1 = shared.settings['name1']
    print('NAME',shared.cha_name)
    name2 = shared.character.replace('_',' ')
    print('ZMY-NAME',name2,"PATH",Path("cache/pfp_character.png").exists())
    visible.reverse()
    for i, _row in enumerate(visible):
        
        if 'name2' in history.keys() and len(history['name2']) > i:
            name2 = history['name2'][i] #i-1  
        else:
            name2 = name2
        print('ZMY-history-name2',name2)
        img_bot = f'<img src="file/cache/pfp_character.png?{name2}">' if Path("cache/pfp_character.png").exists() else ''
        row = [convert_to_markdown(entry) for entry in _row]

        output += f"""
              <div class="message">
                <div class="circle-bot">
                  {img_bot}
                </div>
                <div class="text">
                  <div class="username">
                    {name2}
                  </div>
                  <div class="message-body">
                    {row[1]}
                  </div>
                </div>
              </div>
            """

        if len(row[0]) == 0:  # don't display empty user messages
            continue

        output += f"""
              <div class="message">
                <div class="circle-you">
                  {img_me}
                </div>
                <div class="text">
                  <div class="username">
                    {name1}
                  </div>
                  <div class="message-body">
                    {row[0]}
                  </div>
                </div>
              </div>
            """

    output += "</div>"
    return output

def get_api_chat_result(cha_name,message):
    # print('cha_name')

    history = shared.history['visible']
    print('-'*60)
    print('ZMY_SHARED_HISTORY',shared.history)
    print('-'*60)
    URI = json.load(open('./extensions/api/URI_mapping.json','r'))[cha_name]
    messages = []
    for conv in history[-6:]:
        messages.append({'role':'user','content':conv[0]})
        messages.append({'role':'assistant','content':conv[1]})
    messages.append({'role':'user','content':message})
    print('MESSAGES',messages)
    request = {
        "messages":  messages,#Who do you consider your friends?
        "mode": "chat",
        # "character": "Harry Potter",  
        "your_name": "User", 
        "regenerate": False, 
        "_continue": False, 
        "stop_at_newline": False, 
        "chat_prompt_size": 2048, 
        "chat_generation_attempts": 1, 
        "max_new_tokens": 250, 
        "do_sample": True, 
        "temperature": 0.7, 
        "top_p": 0.1, 
        "typical_p": 1, 
        "epsilon_cutoff": 0, 
        "eta_cutoff": 0, 
        "repetition_penalty": 1.18, 
        "top_k": 40, 
        "min_length": 0, 
        "no_repeat_ngram_size": 0, 
        "num_beams": 1, 
        "penalty_alpha": 0, 
        "length_penalty": 1, 
        "early_stopping": False, 
        "mirostat_mode": 0, 
        "mirostat_tau": 5, 
        "mirostat_eta": 0.1, 
        "seed": -1, 
        "add_bos_token": True, 
        "truncation_length": 2048, 
        "ban_eos_token": False, 
        "skip_special_tokens": True, 
        "stopping_strings": []
        }
    
    response = requests.post(URI, json=request)

    if response.status_code == 200:
        result = response.json()['results'][0]['history']
        # result = response.text()
        print('RESULT',json.loads(response.text))
        print('-'*60)
        print('zmy-result',result)
        print('-'*60)
        # print(json.dumps(result, indent=4))
        # print()
        # print(result)
        shared.reply = result
        return 


def get_api_tweet_result(cha_name,news,tweet_style):
    # print('cha_name')
    URI = json.load(open('./extensions/api/URI_mapping.json','r'))[cha_name]
    request = {
        "news": news,
        "tweet_style": tweet_style,
        "mode": "tweet",
        # "character": "Harry Potter",  
        "your_name": "User", 
        "regenerate": False, 
        "_continue": False, 
        "stop_at_newline": False, 
        "chat_prompt_size": 2048, 
        "chat_generation_attempts": 1, 
        "max_new_tokens": 250, 
        "do_sample": True, 
        "temperature": 0.7, 
        "top_p": 0.1, 
        "typical_p": 1, 
        "epsilon_cutoff": 0, 
        "eta_cutoff": 0, 
        "repetition_penalty": 1.18, 
        "top_k": 40, 
        "min_length": 0, 
        "no_repeat_ngram_size": 0, 
        "num_beams": 1, 
        "penalty_alpha": 0, 
        "length_penalty": 1, 
        "early_stopping": False, 
        "mirostat_mode": 0, 
        "mirostat_tau": 5, 
        "mirostat_eta": 0.1, 
        "seed": -1, 
        "add_bos_token": True, 
        "truncation_length": 2048, 
        "ban_eos_token": False, 
        "skip_special_tokens": True, 
        "stopping_strings": []
        }
    
    response = requests.post(URI, json=request)

    if response.status_code == 200:
        result = response.json()['results'][0]['history']
        # result = response.text()
        print('RESULT',json.loads(response.text))
        print('-'*60)
        print('zmy-result',result)
        print('-'*60)
        # print(json.dumps(result, indent=4))
        # print()
        # print(result)
        shared.reply = result
        name2 = shared.character.replace('_',' ')
        img_bot = f'<img src="file/cache/pfp_character.png?{name2}">' if Path("cache/pfp_character.png").exists() else ''
        output = f"""
              <div class="message">
                <div class="circle-bot">
                  {img_bot}
                </div>
                <div class="text">
                  <div class="username">
                    {name2}
                  </div>
                  <div class="message-body">
                    {result}
                  </div>
                </div>
              </div>
            """
        return output


def get_available_characters():
    cha_ls = json.load(open('./extensions/api/URI_mapping.json','r')).keys()
    return cha_ls

def get_URI(cha_name):
    cha_ls = json.load(open('./extensions/api/URI_mapping.json','r'))
    URI = cha_ls[cha_name]
    return URI


def generate_tweet(news,tweet_style):
    print('ZMY-tweet-style',tweet_style)
    shared.news = news
    prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: Please Generate a tweet style comment in "+tweet_style[0]+" way according to following news content:\n"+news+"\nPlease don't generate tag related to year or specific time. \nASSISTANT:"
    print('ZMY-news',prompt)
    inputs = shared.tokenizer([prompt])
    input_ids = torch.as_tensor(inputs.input_ids).cuda()
    # print(input_ids)
    skip_echo_len = len(prompt)
    # print('ZMY-tweet-model',shared.tweet_model)
    output_ids = shared.tweet_model.generate(
            torch.as_tensor(input_ids).cuda(),
            do_sample=True,
            temperature=0.7,
            max_new_tokens=2048,
        )
    reply = shared.tokenizer.decode(output_ids[0], skip_special_tokens=True)[skip_echo_len:].strip()
    print('ZMY-tweet-reply',reply)
    shared.tweet = reply
    return reply

# def get_turn_substrings(state, instruct=False):
#     if instruct:
#         if 'turn_template' not in state or state['turn_template'] == '':
#             template = '<|user|>\n<|user-message|>\n<|bot|>\n<|bot-message|>\n'
#         else:
#             template = state['turn_template'].replace(r'\n', '\n')
#     else:
#         template = '<|user|>: <|user-message|>\n<|bot|>: <|bot-message|>\n'

#     replacements = {
#         '<|user|>': state['name1_instruct' if instruct else 'name1'].strip(),
#         '<|bot|>': state['name2_instruct' if instruct else 'name2'].strip(),
#     }

#     output = {
#         'user_turn': template.split('<|bot|>')[0],
#         'bot_turn': '<|bot|>' + template.split('<|bot|>')[1],
#         'user_turn_stripped': template.split('<|bot|>')[0].split('<|user-message|>')[0],
#         'bot_turn_stripped': '<|bot|>' + template.split('<|bot|>')[1].split('<|bot-message|>')[0],
#     }

#     for k in output:
#         output[k] = replace_all(output[k], replacements)

#     return output


# def generate_chat_prompt(user_input, state, **kwargs):
#     impersonate = kwargs.get('impersonate', False)
#     _continue = kwargs.get('_continue', False)
#     also_return_rows = kwargs.get('also_return_rows', False)
#     history = kwargs.get('history', shared.history)['internal']
#     is_instruct = state['mode'] == 'instruct'

#     # Find the maximum prompt size
#     chat_prompt_size = state['chat_prompt_size']
#     if shared.soft_prompt:
#         chat_prompt_size -= shared.soft_prompt_tensor.shape[1]

#     max_length = min(get_max_prompt_length(state), chat_prompt_size)
#     all_substrings = {
#         'chat': get_turn_substrings(state, instruct=False),
#         'instruct': get_turn_substrings(state, instruct=True)
#     }

#     substrings = all_substrings['instruct' if is_instruct else 'chat']

#     # Create the template for "chat-instruct" mode
#     if state['mode'] == 'chat-instruct':
#         wrapper = ''
#         command = state['chat-instruct_command'].replace('<|character|>', state['name2'] if not impersonate else state['name1'])
#         wrapper += state['context_instruct']
#         wrapper += all_substrings['instruct']['user_turn'].replace('<|user-message|>', command)
#         wrapper += all_substrings['instruct']['bot_turn_stripped']
#         if impersonate:
#             wrapper += substrings['user_turn_stripped'].rstrip(' ')
#         elif _continue:
#             wrapper += apply_extensions("bot_prefix", substrings['bot_turn_stripped'])
#             wrapper += history[-1][1]
#         else:
#             wrapper += apply_extensions("bot_prefix", substrings['bot_turn_stripped'].rstrip(' '))
#     else:
#         wrapper = '<|prompt|>'

#     # Build the prompt
#     min_rows = 3
#     i = len(history) - 1

#     rows = [state['context_instruct'] if is_instruct else f"{state['context'].strip()}\n"]
#     while i >= 0 and get_encoded_length(wrapper.replace('<|prompt|>', ''.join(rows))) < max_length:
    
#         if _continue and i == len(history) - 1:
#             if state['mode'] != 'chat-instruct':
#                 rows.insert(1, substrings['bot_turn_stripped'] + history[i][1].strip())
#         else:
#             rows.insert(1, substrings['bot_turn'].replace('<|bot-message|>', history[i][1].strip()))

#         string = history[i][0]
#         if string not in ['', '<|BEGIN-VISIBLE-CHAT|>']:
#             rows.insert(1, replace_all(substrings['user_turn'], {'<|user-message|>': string.strip(), '<|round|>': str(i)}))

#         i -= 1

#     if impersonate:
#         if state['mode'] == 'chat-instruct':
#             min_rows = 1
#         else:
#             min_rows = 2
#             rows.append(substrings['user_turn_stripped'].rstrip(' '))
#     elif not _continue:
    
#         # Add the user message
#         if len(user_input) > 0:
#             # print('ZMY-state-name2',state['name2'])
#             # name2 = state['name2'].replace('_',' ')
#             # rows[0] = rows[0].replace('Assistant',f'Assistant named {name2}')
#             rows.append(replace_all(substrings['user_turn'], {'<|user-message|>': user_input.strip(), '<|round|>': str(len(history))}))
        

#         # Add the character prefix
#         if state['mode'] != 'chat-instruct':
#             rows.append(apply_extensions("bot_prefix", substrings['bot_turn_stripped'].rstrip(' ')))
    

#     while len(rows) > min_rows and get_encoded_length(wrapper.replace('<|prompt|>', ''.join(rows))) >= max_length:
#         rows.pop(1)
   
#     print('ZMY-rows',rows)
#     prompt = wrapper.replace('<|prompt|>', ''.join(rows))
#     if also_return_rows:
#         return prompt, rows
#     else:
#         return prompt


# def get_stopping_strings(state):
#     # print('ZMY_STATE',state)
#     stopping_strings = []
#     if state['mode'] in ['instruct', 'chat-instruct']:
#         stopping_strings += [
#             state['turn_template'].split('<|user-message|>')[1].split('<|bot|>')[0] + '<|bot|>',
#             state['turn_template'].split('<|bot-message|>')[1] + '<|user|>'
#         ]

#         replacements = {
#             '<|user|>': state['name1_instruct'],
#             '<|bot|>': state['name2_instruct']
#         }

#         for i in range(len(stopping_strings)):
#             stopping_strings[i] = replace_all(stopping_strings[i], replacements).rstrip(' ').replace(r'\n', '\n')

#     if state['mode'] in ['chat', 'chat-instruct']:
#         stopping_strings += [
#             f"\n{state['name1']}:",
#             f"\n{state['name2']}:"
#         ]

#     stopping_strings += ast.literal_eval(f"[{state['custom_stopping_strings']}]")
#     return stopping_strings


# def extract_message_from_reply(reply, state):
#     next_character_found = False
#     stopping_strings = get_stopping_strings(state)

#     if state['stop_at_newline']:
#         lines = reply.split('\n')
#         reply = lines[0].strip()
#         if len(lines) > 1:
#             next_character_found = True
#     else:
#         for string in stopping_strings:
#             idx = reply.find(string)
#             if idx != -1:
#                 reply = reply[:idx]
#                 next_character_found = True

#         # If something like "\nYo" is generated just before "\nYou:"
#         # is completed, trim it
#         if not next_character_found:
#             for string in stopping_strings:
#                 for j in range(len(string) - 1, 0, -1):
#                     if reply[-j:] == string[:j]:
#                         reply = reply[:-j]
#                         break
#                 else:
#                     continue

#                 break

#     return reply, next_character_found


# def chatbot_wrapper(text, history, state, regenerate=False, _continue=False, loading_message=True):
#     # print('ZMY_DEBUG_text',text)
#     # print('ZMY_DEBUG_history',history)
#     output = copy.deepcopy(history)
#     output = apply_extensions('history', output)
#     if shared.model_name == 'None' or shared.model is None:
#         logger.error("No model is loaded! Select one in the Model tab.")
#         yield output
#         return

#     # Defining some variables
#     just_started = True
#     visible_text = None
#     eos_token = '\n' if state['stop_at_newline'] else None
#     stopping_strings = get_stopping_strings(state)

#     # Preparing the input
#     if not any((regenerate, _continue)):
#         text, visible_text = apply_extensions('input_hijack', text, visible_text)
#         if visible_text is None:
#             visible_text = text

#         text = apply_extensions('input', text)
#         # *Is typing...*
#         if loading_message:
#             yield {'visible': output['visible'] + [[visible_text, shared.processing_message]], 'internal': output['internal']}
#     else:
#         text, visible_text = output['internal'][-1][0], output['visible'][-1][0]
#         if regenerate:
#             output['visible'].pop()
#             output['internal'].pop()
#             # *Is typing...*
#             if loading_message:
#                 yield {'visible': output['visible'] + [[visible_text, shared.processing_message]], 'internal': output['internal']}
#         elif _continue:
#             last_reply = [output['internal'][-1][1], output['visible'][-1][1]]
#             if loading_message:
#                 yield {'visible': output['visible'][:-1] + [[visible_text, last_reply[1] + '...']], 'internal': output['internal']}

#     # Generating the prompt
#     kwargs = {
#         '_continue': _continue,
#         'history': output,
#     }

#     prompt = apply_extensions('custom_generate_chat_prompt', text, state, **kwargs)
#     if prompt is None:
#         prompt = generate_chat_prompt(text, state, **kwargs)

#     print('-'*60)
#     print('ZMY-prompt3',prompt)
#     print('-'*60)
#     #zmy-generate
#     reply = zmy_generate_reply(state['name1'],state['name2'],prompt)
#     # if reply == "":
#     #     print('-'*60)
#     #     print('reloading lora!')
#     #     print('delta_weight_path',delta_weight_path)
#     #     print('-'*60)
#     #     delta_weight_path = yaml.safe_load(open('characters/'+state['name2']+'.yaml','r', encoding='utf-8').read())['delta_weight_path']

#     #     shared.model[state['name2'].replace(' ','_')] = add_lora_to_model(delta_weight_path)
#     #     reply = zmy_generate_reply(state['name2'].replace(' ','_'),prompt)
#     # print('ZMY_DEBUG_reply2',reply)
#     # reply, next_character_found = extract_message_from_reply(reply, state)
#     # visible_reply = re.sub("(<USER>|<user>|{{user}})", state['name1'], reply)
#     # visible_reply = apply_extensions("output", visible_reply)
#     output['visible'].append([text,reply])
#     output['internal'].append([text,reply])
    

#     # # Generate
#     # cumulative_reply = ''
#     # for i in range(state['chat_generation_attempts']):
#     #     reply = None
#     #     for j, reply in enumerate(generate_reply(prompt + cumulative_reply, state, eos_token=eos_token, stopping_strings=stopping_strings, is_chat=True)):
#     #         reply = cumulative_reply + reply

#     #         # Extract the reply
#             # reply, next_character_found = extract_message_from_reply(reply, state)
#             # visible_reply = re.sub("(<USER>|<user>|{{user}})", state['name1'], reply)
#             # visible_reply = apply_extensions("output", visible_reply)

#     #         # We need this global variable to handle the Stop event,
#     #         # otherwise gradio gets confused
#     #         if shared.stop_everything:
#     #             yield output
#     #             return

#     #         if just_started:
#     #             just_started = False
#     #             if not _continue:
#     #                 output['internal'].append(['', ''])
#     #                 output['visible'].append(['', ''])

#     #         if _continue:
#     #             output['internal'][-1] = [text, last_reply[0] + reply]
#     #             output['visible'][-1] = [visible_text, last_reply[1] + visible_reply]
#     #             if state['stream']:
#     #                 yield output
#     #         elif not (j == 0 and visible_reply.strip() == ''):
#     #             output['internal'][-1] = [text, reply.lstrip(' ')]
#     #             output['visible'][-1] = [visible_text, visible_reply.lstrip(' ')]
#     #             if state['stream']:
#     #                 yield output

#     #         if next_character_found:
#     #             break

#     #     if reply in [None, cumulative_reply]:
#     #         break
#     #     else:
#     #         cumulative_reply = reply
#     #zmy
#     for st in ['name1','name2']:
#         if st in output.keys():
#             output[st].append(state[st])
#         else:
#             output[st]=[state[st]]
#     print('ZMY-output',output)
#     yield output


# def impersonate_wrapper(text, state):
#     if shared.model_name == 'None' or shared.model is None:
#         logger.error("No model is loaded! Select one in the Model tab.")
#         yield ''
#         return

#     # Defining some variables
#     cumulative_reply = ''
#     eos_token = '\n' if state['stop_at_newline'] else None
#     prompt = generate_chat_prompt('', state, impersonate=True)
#     stopping_strings = get_stopping_strings(state)

#     yield text + '...'
#     cumulative_reply = text
#     for i in range(state['chat_generation_attempts']):
#         reply = None
#         for reply in generate_reply(prompt + cumulative_reply, state, eos_token=eos_token, stopping_strings=stopping_strings, is_chat=True):
#             reply = cumulative_reply + reply
#             reply, next_character_found = extract_message_from_reply(reply, state)
#             yield reply.lstrip(' ')
#             if shared.stop_everything:
#                 return

#             if next_character_found:
#                 break

#         if reply in [None, cumulative_reply]:
#             break
#         else:
#             cumulative_reply = reply

#     yield cumulative_reply.lstrip(' ')


# def generate_chat_reply(text, history, state, regenerate=False, _continue=False, loading_message=True):
#     # print('ZMY_DEBUG_text',text)
#     # print('ZMY-history',history)
#     if regenerate or _continue:
#         text = ''
#         if (len(history['visible']) == 1 and not history['visible'][0][0]) or len(history['internal']) == 0:
#             yield history
#             return

#     for history in chatbot_wrapper(text, history, state, regenerate=regenerate, _continue=_continue, loading_message=loading_message):
#         yield history
    


# # Same as above but returns HTML
# def generate_chat_reply_wrapper(text, state, regenerate=False, _continue=False):

#     for i, history in enumerate(generate_chat_reply(text, shared.history, state, regenerate, _continue, loading_message=True)):
#         if i != 0:
#             shared.history = copy.deepcopy(history)
#         # print('ZMY-history',history,shared.history)
#         yield chat_html_wrapper(history, state['name1'], state['name2'], state['mode'], state['chat_style'])


# def remove_last_message():
#     if len(shared.history['visible']) > 0 and shared.history['internal'][-1][0] != '<|BEGIN-VISIBLE-CHAT|>':
#         last = shared.history['visible'].pop()
#         shared.history['internal'].pop()
#     else:
#         last = ['', '']

#     return last[0]


# def send_last_reply_to_input():
#     if len(shared.history['internal']) > 0:
#         return shared.history['internal'][-1][1]
#     else:
#         return ''


# def replace_last_reply(text):
#     if len(shared.history['visible']) > 0:
#         shared.history['visible'][-1][1] = text
#         shared.history['internal'][-1][1] = apply_extensions("input", text)


# def send_dummy_message(text):
#     shared.history['visible'].append([text, ''])
#     shared.history['internal'].append([apply_extensions("input", text), ''])


# def send_dummy_reply(text):
#     if len(shared.history['visible']) > 0 and not shared.history['visible'][-1][1] == '':
#         shared.history['visible'].append(['', ''])
#         shared.history['internal'].append(['', ''])

#     shared.history['visible'][-1][1] = text
#     shared.history['internal'][-1][1] = apply_extensions("input", text)


def clear_chat_log(greeting, mode):
    shared.history['visible'] = []
    shared.history['internal'] = []
    shared.history['name1'] = []
    shared.history['name2'] = []

#     if mode != 'instruct':
#         if greeting != '':
#             shared.history['internal'] += [['<|BEGIN-VISIBLE-CHAT|>', greeting]]
#             shared.history['visible'] += [['', apply_extensions("output", greeting)]]

#         save_history(mode)


def redraw_html(name1, name2, mode, style, reset_cache=False):
    shared.history = {'internal':[],'visible':[]}
    return generate_cai_chat_html()


# def tokenize_dialogue(dialogue, name1, name2):
#     history = []
#     messages = []
#     dialogue = re.sub('<START>', '', dialogue)
#     dialogue = re.sub('<start>', '', dialogue)
#     dialogue = re.sub('(\n|^)[Aa]non:', '\\1You:', dialogue)
#     dialogue = re.sub('(\n|^)\[CHARACTER\]:', f'\\g<1>{name2}:', dialogue)
#     idx = [m.start() for m in re.finditer(f"(^|\n)({re.escape(name1)}|{re.escape(name2)}):", dialogue)]
#     if len(idx) == 0:
#         return history

#     for i in range(len(idx) - 1):
#         messages.append(dialogue[idx[i]:idx[i + 1]].strip())

#     messages.append(dialogue[idx[-1]:].strip())
#     entry = ['', '']
#     for i in messages:
#         if i.startswith(f'{name1}:'):
#             entry[0] = i[len(f'{name1}:'):].strip()
#         elif i.startswith(f'{name2}:'):
#             entry[1] = i[len(f'{name2}:'):].strip()
#             if not (len(entry[0]) == 0 and len(entry[1]) == 0):
#                 history.append(entry)

#             entry = ['', '']

#     print("\033[1;32;1m\nDialogue tokenized to:\033[0;37;0m\n", end='')
#     for row in history:
#         for column in row:
#             print("\n")
#             for line in column.strip().split('\n'):
#                 print("|  " + line + "\n")

#             print("|\n")
#         print("------------------------------")

#     return history


# # def save_history(mode, timestamp=False):
# #     # Instruct mode histories should not be saved as if
# #     # Alpaca or Vicuna were characters
# #     if mode == 'instruct':
# #         if not timestamp:
# #             return

# #         fname = f"Instruct_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
# #     else:
# #         if timestamp:
# #             fname = f"{shared.character}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
# #         else:
# #             fname = f"{shared.character}_persistent.json"

# #     if not Path('logs').exists():
# #         Path('logs').mkdir()

# #     with open(f'logs/{fname}', 'w') as f:
# #     # with open(Path(f'logs/{fname}'), 'w', encoding='utf-8') as f:
# #         f.write(json.dumps({'data': shared.history['internal'], 'data_visible': shared.history['visible']}, indent=2))

# #     return Path(f'logs/{fname}')

def save_history(user_input):
    shared.history['visible'].append([user_input,shared.reply])
    # print("HISTORY",shared.history)
    return 


# def load_history(file, name1, name2):
#     file = file.decode('utf-8')
#     try:
#         j = json.loads(file)
#         if 'data' in j:
#             shared.history['internal'] = j['data']
#             if 'data_visible' in j:
#                 shared.history['visible'] = j['data_visible']
#             else:
#                 shared.history['visible'] = copy.deepcopy(shared.history['internal'])
#     except:
#         shared.history['internal'] = tokenize_dialogue(file, name1, name2)
#         shared.history['visible'] = copy.deepcopy(shared.history['internal'])


# def replace_character_names(text, name1, name2):
#     text = text.replace('{{user}}', name1).replace('{{char}}', name2)
#     return text.replace('<USER>', name1).replace('<BOT>', name2)


# def build_pygmalion_style_context(data):
#     context = ""
#     if 'char_persona' in data and data['char_persona'] != '':
#         context += f"{data['char_name']}'s Persona: {data['char_persona']}\n"

#     if 'world_scenario' in data and data['world_scenario'] != '':
#         context += f"Scenario: {data['world_scenario']}\n"

#     context = f"{context.strip()}\n<START>\n"
#     return context

def make_thumbnail(image):
    image = image.resize((350, round(image.size[1] / image.size[0] * 350)), Image.Resampling.LANCZOS)
    if image.size[1] > 470:
        image = ImageOps.fit(image, (350, 470), Image.ANTIALIAS)

    return image

def generate_pfp_cache(character):
    cache_folder = Path("cache")
    if not cache_folder.exists():
        cache_folder.mkdir()

    for path in [Path(f"characters/{character}.{extension}") for extension in ['png', 'jpg', 'jpeg']]:
        if path.exists():
            img = make_thumbnail(Image.open(path))
            img.save(Path('cache/pfp_character.png'), format='PNG')
            return img

    return None

# def load_your_name(name):
#     return name

def load_character(character, name1, name2, instruct=False):
    # print('ZMY-character',character)
    shared.character = character.replace(' ','_')
    print('CHA',shared.character)
    # context = greeting = turn_template = ""
    # greeting_field = 'greeting'
    picture = None

    # Deleting the profile picture cache, if any
    if Path("cache/pfp_character.png").exists():
        Path("cache/pfp_character.png").unlink()

    if character != 'None': #'None'
        folder = 'characters' if not instruct else 'characters/instruction-following'
        picture = generate_pfp_cache(character)
        for extension in ["yml", "yaml", "json"]:
            filepath = Path(f'{folder}/{character}.{extension}')
            if filepath.exists():
                break
    print('CHA',character,"picture",picture)

    #     file_contents = open(filepath, 'r', encoding='utf-8').read()
    #     data = json.loads(file_contents) if extension == "json" else yaml.safe_load(file_contents)

    #     # Finding the bot's name
    #     for k in ['name', 'bot', '<|bot|>', 'char_name']:
    #         if k in data and data[k] != '':
    #             name2 = data[k]
    #             break

    #     # Find the user name (if any)
    #     for k in ['your_name', 'user', '<|user|>']:
    #         if k in data and data[k] != '':
    #             name1 = data[k]
    #             break

    #     for field in ['context', 'greeting', 'example_dialogue', 'char_persona', 'char_greeting', 'world_scenario']:
    #         if field in data:
    #             data[field] = replace_character_names(data[field], name1, name2)

    #     if 'context' in data:
    #         context = data['context']
    #         if not instruct:
    #             context = context.strip() + '\n'
    #     elif "char_persona" in data:
    #         context = build_pygmalion_style_context(data)
    #         greeting_field = 'char_greeting'

    #     if 'example_dialogue' in data:
    #         context += f"{data['example_dialogue'].strip()}\n"

    #     if greeting_field in data:
    #         greeting = data[greeting_field]

    #     if 'turn_template' in data:
    #         turn_template = data['turn_template']
        
    #     # if 'delta_weight_path' in data:
    #     #     delta_weight_path = data['delta_weight_path']
    #     # else:
    #     #     raise ValueError("Please provide character delta weight path!")

    # else:
    #     context = shared.settings['context']
    #     name2 = shared.settings['name2']
    #     greeting = shared.settings['greeting']
    #     turn_template = shared.settings['turn_template']

    # print('ZMY-name2',name2)
    # # print('ZMY10',delta_weight_path)
    # # if name2.replace(' ','_') not in shared.model.keys():
    # #     print('loading lora')
    # #     shared.model[name2.replace(' ','_')] = add_lora_to_model(delta_weight_path)
    # #     print(shared.model[name2.replace(' ','_')])
    # # else:
    # #     print("lora has been loaded before!")

    # if not instruct:
    #     shared.history['internal'] = []
    #     shared.history['visible'] = []
    #     if Path(f'logs/{shared.character}_persistent.json').exists():
    #         load_history(open(Path(f'logs/{shared.character}_persistent.json'), 'rb').read(), name1, name2)
    #     else:
    #         # Insert greeting if it exists
    #         if greeting != "":
    #             shared.history['internal'] += [['<|BEGIN-VISIBLE-CHAT|>', greeting]]
    #             shared.history['visible'] += [['', apply_extensions("output", greeting)]]

    #         # Create .json log files since they don't already exist
    #         save_history('instruct' if instruct else 'chat')

    return picture# name1, name2, picture, greeting, context, repr(turn_template)[1:-1]#, delta_weight_path


# @functools.cache
# def load_character_memoized(character, name1, name2, instruct=False):
#     # print('ZMY-cha2',character)
#     return load_character(character, name1, name2, instruct=instruct)


# def upload_character(json_file, img, tavern=False):
#     json_file = json_file if type(json_file) == str else json_file.decode('utf-8')
#     data = json.loads(json_file)
#     outfile_name = data["char_name"]
#     i = 1
#     while Path(f'characters/{outfile_name}.json').exists():
#         outfile_name = f'{data["char_name"]}_{i:03d}'
#         i += 1

#     if tavern:
#         outfile_name = f'TavernAI-{outfile_name}'

#     with open(Path(f'characters/{outfile_name}.json'), 'w', encoding='utf-8') as f:
#         f.write(json_file)

#     if img is not None:
#         img = Image.open(io.BytesIO(img))
#         img.save(Path(f'characters/{outfile_name}.png'))

#     logger.info(f'New character saved to "characters/{outfile_name}.json".')
#     return outfile_name


# def upload_tavern_character(img, name1, name2):
#     _img = Image.open(io.BytesIO(img))
#     _img.getexif()
#     decoded_string = base64.b64decode(_img.info['chara'])
#     _json = json.loads(decoded_string)
#     _json = {"char_name": _json['name'], "char_persona": _json['description'], "char_greeting": _json["first_mes"], "example_dialogue": _json['mes_example'], "world_scenario": _json['scenario']}
#     return upload_character(json.dumps(_json), img, tavern=True)


def upload_your_profile_picture(img):
    cache_folder = Path("cache")
    if not cache_folder.exists():
        cache_folder.mkdir()

    if img is None:
        if Path("cache/pfp_me.png").exists():
            Path("cache/pfp_me.png").unlink()
    else:
        img = make_thumbnail(img)
        img.save(Path('cache/pfp_me.png'))
        logger.info('Profile picture saved to "cache/pfp_me.png"')


# def delete_file(path):
#     if path.exists():
#         logger.warning(f'Deleting {path}')
#         path.unlink(missing_ok=True)


# def save_character(name, greeting, context, picture, filename, instruct=False):
#     if filename == "":
#         logger.error("The filename is empty, so the character will not be saved.")
#         return

#     folder = 'characters' if not instruct else 'characters/instruction-following'
#     data = {
#         'name': name,
#         'greeting': greeting,
#         'context': context,
#     }

#     data = {k: v for k, v in data.items() if v}  # Strip falsy
#     filepath = Path(f'{folder}/{filename}.yaml')
#     with filepath.open('w') as f:
#         yaml.dump(data, f, sort_keys=False)

#     logger.info(f'Wrote {filepath}')
#     path_to_img = Path(f'{folder}/{filename}.png')
#     if picture and not instruct:
#         picture.save(path_to_img)
#         logger.info(f'Wrote {path_to_img}')
#     elif path_to_img.exists():
#         delete_file(path_to_img)


# def delete_character(name, instruct=False):
#     folder = 'characters' if not instruct else 'characters/instruction-following'
#     for extension in ["yml", "yaml", "json"]:
#         delete_file(Path(f'{folder}/{name}.{extension}'))

#     delete_file(Path(f'{folder}/{name}.png'))
