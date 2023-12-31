import requests

# For local streaming, the websockets are hosted without ssl - http://
HOST = 'localhost:5000'
URI = f'http://{HOST}/api/v1/chat'

# For reverse-proxied streaming, the remote will likely host with ssl - https://
# URI = 'https://your-uri-here.trycloudflare.com/api/v1/generate'


def run(prompt):
    request = {
        'prompt': prompt,
        'max_new_tokens': 250,
        'do_sample': True,
        'temperature': 1.3,
        'top_p': 0.1,
        'typical_p': 1,
        'epsilon_cutoff': 0,  # In units of 1e-4
        'eta_cutoff': 0,  # In units of 1e-4
        'repetition_penalty': 1.18,
        'top_k': 40,
        'min_length': 0,
        'no_repeat_ngram_size': 0,
        'num_beams': 1,
        'penalty_alpha': 0,
        'length_penalty': 1,
        'early_stopping': False,
        'mirostat_mode': 0,
        'mirostat_tau': 5,
        'mirostat_eta': 0.1,
        'seed': -1,
        'add_bos_token': True,
        'truncation_length': 2048,
        'ban_eos_token': False,
        'skip_special_tokens': True,
        'stopping_strings': [],
        'mem_root': "./extensions/long_term_memory/user_data/bot_memories_zmy/",
        'history': {'internal': [], 'visible': []},
        'your_name': 'You',
        'context': """ This is a conversation with your Assistant. It is a computer program designed to help you with various tasks such as answering questions, providing recommendations,  and helping with decision making. You can ask it anything you want and it will do its best to give you accurate and relevant information. """
    }

    response = requests.post(URI, json=request)

    if response.status_code == 200:
        result = response.json()['results'][0]['history']['visible'][0][1]
        print(result)


if __name__ == '__main__':
    print("INPUT:")
    prompt = input()
    print("ASSISTANT OUTPUT:")
    run(prompt)
