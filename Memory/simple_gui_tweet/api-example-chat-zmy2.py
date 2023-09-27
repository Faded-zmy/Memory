import json

import requests

# For local streaming, the websockets are hosted without ssl - http://
# HOST = 'localhost:5000'
HOST = '35.92.206.66:3000'
URI = f'http://{HOST}/api/v1/chat'
# URI = "http://dev-ai-chat-character-service.dadao.works/api/v1/chat"

# For reverse-proxied streaming, the remote will likely host with ssl - https://
# URI = 'https://your-uri-here.trycloudflare.com/api/v1/generate'


def run():
    # request = {
    #     'messages':[
    #         # {"role": "system", "content": 'This is a conversation with your Assistant. The Assistant is very helpful and is eager to chat with you and answer your questions.'},
    #         {"role": "user", "content": "Who won the world series in 2020?"},
    #         {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
    #         # {"role": "user", "content": "Where was it played?"}
    #         {'role':'user','content':'what is your name?'}
    #     ],
    #     # 'user_input': user_input,
    #     # 'history': history,
    #     'mode': 'chat',  # Valid options: 'chat', 'chat-instruct', 'instruct'
    #     'character': 'Harry_Potter',
    #     # 'instruction_template': 'Open Assistant',
    #     'your_name': 'mengying_new',

    #     'regenerate': False,
    #     '_continue': False,
    #     'stop_at_newline': False,
    #     'chat_prompt_size': 2048,
    #     'chat_generation_attempts': 1,
    #     # 'chat-instruct_command': 'Continue the chat dialogue below. Write a single reply for the character "<|character|>".\n\n<|prompt|>',

    #     'max_new_tokens': 250,
    #     'do_sample': True,
    #     'temperature': 0.7,
    #     'top_p': 0.1,
    #     'typical_p': 1,
    #     'epsilon_cutoff': 0,  # In units of 1e-4
    #     'eta_cutoff': 0,  # In units of 1e-4
    #     'repetition_penalty': 1.18,
    #     'top_k': 40,
    #     'min_length': 0,
    #     'no_repeat_ngram_size': 0,
    #     'num_beams': 1,
    #     'penalty_alpha': 0,
    #     'length_penalty': 1,
    #     'early_stopping': False,
    #     'mirostat_mode': 0,
    #     'mirostat_tau': 5,
    #     'mirostat_eta': 0.1,
    #     'seed': -1,
    #     'add_bos_token': True,
    #     'truncation_length': 2048,
    #     'ban_eos_token': False,
    #     'skip_special_tokens': True,
    #     'stopping_strings': [],
    #     # 'mem_root': "./extensions/long_term_memory/user_data/bot_memories_Sherlock/",
    # }
    request = {
        "messages":  [{'role': 'user', 'content': "Who do you consider your friends?"}],#Who do you consider your friends?
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
        # print('zmy-result',result)
        # print(json.dumps(result, indent=4))
        print()
        print(result['visible'][-1][1])
        return result
    


if __name__ == '__main__':

    # user_input = "Please give me a step-by-step guide on how to plant a tree in my backyard."

    # Basic example
    result = run()
    # history = {'internal': [], 'visible': []}

    # # "Continue" example. Make sure to set '_continue' to True above
    # # arr = [user_input, 'Surely, here is']
    # # history = {'internal': [arr], 'visible': [arr]}

    # print('If you want to end the conversation,input "END"!')
    # while True:
    #     print("HISTORY:",history)
    #     print("INPUT:")
    #     user_input = input()
    #     if user_input == "END":
    #         break
    #     else:
    #         print('ASSISTANT:')
    #         result = run(user_input, history)
    #         if len(history['internal'])< 3:
    #             history['internal'].append(result['internal'][-1])
    #             history['visible'].append(result['visible'][-1])
    #         else:
    #             history['internal'] = [result['internal'][-1]]
    #             history['visible'] = [result['visible'][-1]]

            

