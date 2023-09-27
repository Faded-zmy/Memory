import ast
import random
import re
import threading
import time
import traceback

import numpy as np
import torch
import transformers

import modules.shared as shared
# from modules.callbacks import (Iteratorize, Stream,
#                                _SentinelTokenStoppingCriteria)
from modules.extensions import apply_extensions
from modules.html_generator import generate_4chan_html, generate_basic_html
from extensions.long_term_memory.script import custom_generate_chat_prompt
# from modules.logging_colors import logger
# from modules.models import clear_torch_cache, local_rank


def generate_tweet(news,tweet_style):
    print('ZMY-tweet-style',tweet_style)
    shared.news = news
    prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: Please Generate a tweet style comment in "+tweet_style[0][0]+" way according to following news content:\n"+news+"\nPlease don't generate tag related to year or specific time. \nASSISTANT:"
    print('ZMY-news',prompt)
    inputs = shared.tokenizer([prompt])
    input_ids = torch.as_tensor(inputs.input_ids).cuda()
    # print(input_ids)
    skip_echo_len = len(prompt)
    # print('ZMY-tweet-model',shared.tweet_model)
    # output_ids = shared.tweet_model.generate(
    output_ids = shared.model.generate(
            torch.as_tensor(input_ids).cuda(),
            do_sample=True,
            temperature=0.7,
            max_new_tokens=2048,
        )
    reply = shared.tokenizer.decode(output_ids[0], skip_special_tokens=True)[skip_echo_len:].strip()
    print('ZMY-tweet-reply',reply)
    shared.tweet = reply
    return reply


def zmy_encode(prompt):
    inputs = shared.tokenizer([prompt])
    input_ids=torch.as_tensor(inputs.input_ids).cuda()
    skip_echo_len = len(prompt.replace('</s>',''))# - prompt.count("</s>") * 4
    return input_ids,skip_echo_len


def zmy_decode(output_ids, skip_special_tokens=True):
    return shared.tokenizer.decode(output_ids, skip_special_tokens)
def generate_chat_reply(name1,name2,history,prompt):
    
    # prompt=f"A chat between a curious user and an artificial intelligence assistant named {name2.replace('_',' ')}. The assistant gives helpful, detailed, and polite answers to the user's questions. \n"+prompt#.replace(name1,'USER').replace(name2,'ASSISTANT')
    # prompt="A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: "+prompt+" ASSISTANT:"
    # prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: Can you tell me your name? ASSISTANT:"
    shared.history = {'internal':[],'visible':history}
    print('ZMY#_PROMPT',prompt,shared.history)
    if 'long_term_memory' in shared.args.extensions:
        mem_prompt = custom_generate_chat_prompt(prompt)
        prompt = mem_prompt+'\n'+prompt
        if mem_prompt == '':
            prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."+prompt
    print('ZMY-generate-prompt2',prompt)
    input_ids,skip_echo_len = zmy_encode(prompt)
    # print('ZMY_DEBUG-model-key',shared.model.keys())
    print('ZMY_NAME2',name2)
    # model = getattr(shared,name2.replace(' ','_'))
    # model = shared.model[name2]
    output_ids = shared.model.generate(
            torch.as_tensor(input_ids).cuda(),
            do_sample=True,
            temperature=0.7,
            max_new_tokens=2048,
        )
    # outputs = shared.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    # print('ZMY-output_ids',output_ids)
    reply = zmy_decode(output_ids[0], skip_special_tokens=True)[skip_echo_len:].strip()#zmy
    print('-'*60)
    print('ZMY-generate-reply',reply)
    print('-'*60)
    return reply


# def generate_reply(*args, **kwargs):
#     shared.generation_lock.acquire()
#     try:
#         for result in _generate_reply(*args, **kwargs):
#             yield result
#     finally:
#         shared.generation_lock.release()


# def get_max_prompt_length(state):
#     max_length = state['truncation_length'] - state['max_new_tokens']
#     if shared.soft_prompt:
#         max_length -= shared.soft_prompt_tensor.shape[1]

#     return max_length


# def encode(prompt, add_special_tokens=True, add_bos_token=True, truncation_length=None):
#     if shared.model_type in ['rwkv', 'llamacpp']:
#         input_ids = shared.tokenizer.encode(str(prompt))
#         input_ids = np.array(input_ids).reshape(1, len(input_ids))
#         return input_ids
#     else:
#         input_ids = shared.tokenizer.encode(str(prompt), return_tensors='pt', add_special_tokens=add_special_tokens)

#         # This is a hack for making replies more creative.
#         if not add_bos_token and input_ids[0][0] == shared.tokenizer.bos_token_id:
#             input_ids = input_ids[:, 1:]

#         # Llama adds this extra token when the first character is '\n', and this
#         # compromises the stopping criteria, so we just remove it
#         if type(shared.tokenizer) is transformers.LlamaTokenizer and input_ids[0][0] == 29871:
#             input_ids = input_ids[:, 1:]

#     # Handling truncation
#     if truncation_length is not None:
#         input_ids = input_ids[:, -truncation_length:]

#     if shared.model_type in ['rwkv', 'llamacpp'] or shared.args.cpu:
#         return input_ids
#     elif shared.args.flexgen:
#         return input_ids.numpy()
#     elif shared.args.deepspeed:
#         return input_ids.to(device=local_rank)
#     elif torch.has_mps:
#         device = torch.device('mps')
#         return input_ids.to(device)
#     else:
#         return input_ids.cuda()


# def get_encoded_length(prompt):
#     length_after_extensions = apply_extensions('tokenized_length', prompt)
#     if length_after_extensions is not None:
#         return length_after_extensions

#     return len(encode(prompt)[0])


# def decode(output_ids, skip_special_tokens=True):
#     return shared.tokenizer.decode(output_ids, skip_special_tokens)


# def generate_softprompt_input_tensors(input_ids):
#     inputs_embeds = shared.model.transformer.wte(input_ids)
#     inputs_embeds = torch.cat((shared.soft_prompt_tensor, inputs_embeds), dim=1)
#     filler_input_ids = torch.zeros((1, inputs_embeds.shape[1]), dtype=input_ids.dtype).to(shared.model.device)
#     # filler_input_ids += shared.model.config.bos_token_id # setting dummy input_ids to bos tokens
#     return inputs_embeds, filler_input_ids


# # Removes empty replies from gpt4chan outputs
# def fix_gpt4chan(s):
#     for i in range(10):
#         s = re.sub("--- [0-9]*\n>>[0-9]*\n---", "---", s)
#         s = re.sub("--- [0-9]*\n *\n---", "---", s)
#         s = re.sub("--- [0-9]*\n\n\n---", "---", s)

#     return s


# # Fix the LaTeX equations in galactica
# def fix_galactica(s):
#     s = s.replace(r'\[', r'$')
#     s = s.replace(r'\]', r'$')
#     s = s.replace(r'\(', r'$')
#     s = s.replace(r'\)', r'$')
#     s = s.replace(r'$$', r'$')
#     s = re.sub(r'\n', r'\n\n', s)
#     s = re.sub(r"\n{3,}", "\n\n", s)
#     return s


# def get_reply_from_output_ids(output_ids, input_ids, original_question, state, is_chat=False):
#     if shared.model_type == 'HF_seq2seq':
#         reply = decode(output_ids, state['skip_special_tokens'])
#     else:
#         new_tokens = len(output_ids) - len(input_ids[0])
#         reply = decode(output_ids[-new_tokens:], state['skip_special_tokens'])

#         # Prevent LlamaTokenizer from skipping a space
#         if type(shared.tokenizer) is transformers.LlamaTokenizer and len(output_ids) > 0:
#             if shared.tokenizer.convert_ids_to_tokens(int(output_ids[-new_tokens])).startswith('▁'):
#                 reply = ' ' + reply

#     if not is_chat:
#         reply = apply_extensions('output', reply)
#     print('ZMY-reply',reply)

#     return reply


# def formatted_outputs(reply, model_name):
#     if shared.model_type == 'gpt4chan':
#         reply = fix_gpt4chan(reply)
#         return reply, generate_4chan_html(reply)
#     else:
#         return reply, generate_basic_html(reply)


# def set_manual_seed(seed):
#     seed = int(seed)
#     if seed == -1:
#         seed = random.randint(1, 2**31)

#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)

#     return seed


# def stop_everything_event():
#     shared.stop_everything = True


# def generate_reply_wrapper(question, state, eos_token=None, stopping_strings=None):
#     for reply in generate_reply(question, state, eos_token, stopping_strings, is_chat=False):
#         if shared.model_type not in ['HF_seq2seq']:
#             reply = question + reply

#         yield formatted_outputs(reply, shared.model_name)


# def _generate_reply(question, state, eos_token=None, stopping_strings=None, is_chat=False):
#     # print('ZMY',state['delta_weight_path'])
#     state = apply_extensions('state', state)
#     generate_func = apply_extensions('custom_generate_reply')
#     if generate_func is None:
#         if shared.model_name == 'None' or shared.model is None:
#             logger.error("No model is loaded! Select one in the Model tab.")
#             yield ''
#             return

#         if shared.model_type in ['rwkv', 'llamacpp']:
#             generate_func = generate_reply_custom
#         elif shared.args.flexgen:
   
#             generate_func = generate_reply_flexgen
#         else:

#             generate_func = generate_reply_HF

#     # Preparing the input
#     original_question = question
#     if not is_chat:
#         question = apply_extensions('input', question)

#     if shared.args.verbose:
#         print(f'\n\n{question}\n--------------------\n')

#     shared.stop_everything = False
#     clear_torch_cache()
#     seed = set_manual_seed(state['seed'])
#     print('zmy-prompt',question)
#     for reply in generate_func(question, original_question, seed, state, eos_token, stopping_strings, is_chat=is_chat):
#         yield reply
  


# def generate_reply_HF(question, original_question, seed, state, eos_token=None, stopping_strings=None, is_chat=False):
#     print('HF-question',question)
#     generate_params = {}
#     for k in ['max_new_tokens', 'do_sample', 'temperature', 'top_p', 'typical_p', 'repetition_penalty', 'encoder_repetition_penalty', 'top_k', 'min_length', 'no_repeat_ngram_size', 'num_beams', 'penalty_alpha', 'length_penalty', 'early_stopping']:
#         generate_params[k] = state[k]

#     for k in ['epsilon_cutoff', 'eta_cutoff']:
#         if state[k] > 0:
#             generate_params[k] = state[k] * 1e-4

#     if state['ban_eos_token']:
#         generate_params['suppress_tokens'] = [shared.tokenizer.eos_token_id]

#     if shared.args.no_cache:
#         generate_params.update({'use_cache': False})

#     if shared.args.deepspeed:
#         generate_params.update({'synced_gpus': True})

#     # Encode the input
 
#     input_ids = encode(question, add_bos_token=state['add_bos_token'], truncation_length=get_max_prompt_length(state))
#     output = input_ids[0]
#     cuda = not any((shared.args.cpu, shared.args.deepspeed))

#     # Find the eos tokens
#     eos_token_ids = [shared.tokenizer.eos_token_id] if shared.tokenizer.eos_token_id is not None else []
#     if eos_token is not None:
#         eos_token_ids.append(int(encode(eos_token)[0][-1]))

#     # Add the encoded tokens to generate_params
#     if shared.soft_prompt:
#         inputs_embeds, filler_input_ids = generate_softprompt_input_tensors(input_ids)
#         question, filler_input_ids, inputs_embeds = apply_extensions('tokenizer', state, question, filler_input_ids, inputs_embeds)
#         original_input_ids = input_ids
#         generate_params.update({'inputs_embeds': inputs_embeds})
#         generate_params.update({'inputs': filler_input_ids})
#     else:
#         question, input_ids, inputs_embeds = apply_extensions('tokenizer', state, question, input_ids, None)
#         original_input_ids = input_ids
#         generate_params.update({'inputs': input_ids})
#         if inputs_embeds is not None:
#             generate_params.update({'inputs_embeds': inputs_embeds})

#     # Create the StoppingCriteriaList with the stopping strings (needs to be done after tokenizer extensions)
#     stopping_criteria_list = transformers.StoppingCriteriaList()
#     for st in (stopping_strings, ast.literal_eval(f"[{state['custom_stopping_strings']}]")):
#         if type(st) is list and len(st) > 0:
#             sentinel_token_ids = [encode(string, add_special_tokens=False) for string in st]
#             stopping_criteria_list.append(_SentinelTokenStoppingCriteria(sentinel_token_ids=sentinel_token_ids, starting_idx=len(input_ids[0])))
#             break

#     # Update generate_params with the eos token and the stopping strings
#     generate_params['eos_token_id'] = eos_token_ids
#     generate_params['stopping_criteria'] = stopping_criteria_list

#     t0 = time.time()
#     try:
#         if not is_chat and shared.model_type != 'HF_seq2seq':
#             print("ZMY1")
#             yield ''

#         # Generate the entire reply at once.
#         if not state['stream']:
#             print('ZMY1-name2',state['name2'])
#             with torch.no_grad():
#                 output = shared.model.generate(**generate_params)[0]
#                 if cuda:
#                     output = output.cuda()

#             if shared.soft_prompt:
#                 output = torch.cat((input_ids[0], output[filler_input_ids.shape[1]:]))

#             yield get_reply_from_output_ids(output, input_ids, original_question, state, is_chat=is_chat)

#         # Stream the reply 1 token at a time.
#         # This is based on the trick of using 'stopping_criteria' to create an iterator.
#         else:
#             print('ZMY0-name2',state['name2'].replace(' ','_'))
#             print('ZMY-model-key',shared.model.keys())
#             def generate_with_callback(callback=None, **kwargs):
#                 kwargs['stopping_criteria'].append(Stream(callback_func=callback))
#                 clear_torch_cache()
#                 with torch.no_grad():
#                     shared.model[state['name2'].replace(' ','_')].generate(**kwargs)

#             def generate_with_streaming(**kwargs):
#                 return Iteratorize(generate_with_callback, kwargs, callback=None)

#             with generate_with_streaming(**generate_params) as generator:
#                 for output in generator:
#                     if shared.soft_prompt:
#                         output = torch.cat((input_ids[0], output[filler_input_ids.shape[1]:]))

#                     yield get_reply_from_output_ids(output, input_ids, original_question, state, is_chat=is_chat)
#                     if output[-1] in eos_token_ids:
#                         break
#             # print('ZMY-ouput',output)

#     except Exception:
#         print('ZMY-ERROR')
#         traceback.print_exc()
#     finally:
#         t1 = time.time()
#         original_tokens = len(original_input_ids[0])
#         new_tokens = len(output) - (original_tokens if shared.model_type != 'HF_seq2seq' else 0)
#         print(f'Output generated in {(t1-t0):.2f} seconds ({new_tokens/(t1-t0):.2f} tokens/s, {new_tokens} tokens, context {original_tokens}, seed {seed})')
#         return


# def generate_reply_custom(question, original_question, seed, state, eos_token=None, stopping_strings=None, is_chat=False):
#     seed = set_manual_seed(state['seed'])
#     generate_params = {'token_count': state['max_new_tokens']}
#     for k in ['temperature', 'top_p', 'top_k', 'repetition_penalty']:
#         generate_params[k] = state[k]

#     if shared.model_type == 'llamacpp':
#         for k in ['mirostat_mode', 'mirostat_tau', 'mirostat_eta']:
#             generate_params[k] = state[k]

#     t0 = time.time()
#     reply = ''
#     try:
#         if not is_chat:
#             yield ''

#         if not state['stream']:
#             reply = shared.model.generate(context=question, **generate_params)
#             if not is_chat:
#                 reply = apply_extensions('output', reply)

#             yield reply
#         else:
#             for reply in shared.model.generate_with_streaming(context=question, **generate_params):
#                 if not is_chat:
#                     reply = apply_extensions('output', reply)

#                 yield reply

#     except Exception:
#         traceback.print_exc()
#     finally:
#         t1 = time.time()
#         original_tokens = len(encode(original_question)[0])
#         new_tokens = len(encode(original_question + reply)[0]) - original_tokens
#         print(f'Output generated in {(t1-t0):.2f} seconds ({new_tokens/(t1-t0):.2f} tokens/s, {new_tokens} tokens, context {original_tokens}, seed {seed})')
#         return


# def generate_reply_flexgen(question, original_question, seed, state, eos_token=None, stopping_strings=None, is_chat=False):
#     generate_params = {}
#     for k in ['max_new_tokens', 'do_sample', 'temperature']:
#         generate_params[k] = state[k]

#     if state['stream']:
#         generate_params['max_new_tokens'] = 8

#     # Encode the input
#     input_ids = encode(question, add_bos_token=state['add_bos_token'], truncation_length=get_max_prompt_length(state))
#     output = input_ids[0]

#     # Find the eos tokens
#     eos_token_ids = [shared.tokenizer.eos_token_id] if shared.tokenizer.eos_token_id is not None else []
#     if eos_token is not None:
#         eos_token_ids.append(int(encode(eos_token)[0][-1]))

#     # Add the encoded tokens to generate_params
#     question, input_ids, inputs_embeds = apply_extensions('tokenizer', state, question, input_ids, None)
#     original_input_ids = input_ids
#     generate_params.update({'inputs': input_ids})
#     if inputs_embeds is not None:
#         generate_params.update({'inputs_embeds': inputs_embeds})

#     # Update generate_params with the eos token and the stopping strings
#     generate_params['stop'] = eos_token_ids[-1]

#     t0 = time.time()
#     try:
#         if not is_chat:
#             yield ''

#         # Generate the entire reply at once.
#         if not state['stream']:
#             with torch.no_grad():
#                 output = shared.model.generate(**generate_params)[0]

#             yield get_reply_from_output_ids(output, input_ids, original_question, state, is_chat=is_chat)

#         # Stream the output naively for FlexGen since it doesn't support 'stopping_criteria'
#         else:
#             for i in range(state['max_new_tokens'] // 8 + 1):
#                 if shared.stop_everything:
#                     break

#                 clear_torch_cache()
#                 with torch.no_grad():
#                     output = shared.model.generate(**generate_params)[0]

#                 if np.count_nonzero(np.isin(input_ids[0], eos_token_ids)) < np.count_nonzero(np.isin(output, eos_token_ids)):
#                     break

#                 yield get_reply_from_output_ids(output, original_input_ids, original_question, state)
#                 input_ids = np.reshape(output, (1, output.shape[0]))
#                 generate_params.update({'inputs': input_ids})

#     except Exception:
#         traceback.print_exc()
#     finally:
#         t1 = time.time()
#         original_tokens = len(original_input_ids[0])
#         new_tokens = len(output) - (original_tokens if shared.model_type != 'HF_seq2seq' else 0)
#         print(f'Output generated in {(t1-t0):.2f} seconds ({new_tokens/(t1-t0):.2f} tokens/s, {new_tokens} tokens, context {original_tokens}, seed {seed})')
#         return
