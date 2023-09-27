import os
import warnings
import yaml
import requests

# from modules.logging_colors import logger

os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
os.environ['BITSANDBYTES_NOWELCOME'] = '1'
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')


# This is a hack to prevent Gradio from phoning home when it gets imported
def my_get(url, **kwargs):
    logger.info('Gradio HTTP request redirected to localhost :)')
    kwargs.setdefault('allow_redirects', True)
    return requests.api.request('get', 'http://127.0.0.1/', **kwargs)


original_get = requests.get
requests.get = my_get
import gradio as gr
requests.get = original_get

import matplotlib
matplotlib.use('Agg')  # This fixes LaTeX rendering on some systems

import importlib
import io
import json
import math
import os
import re
import sys
import time
import traceback
import zipfile
from datetime import datetime
from functools import partial
from pathlib import Path
from threading import Lock

import psutil
import torch
import yaml
from PIL import Image

import modules.extensions as extensions_module
from modules import chat, shared, ui, utils
from modules.extensions import apply_extensions
# from modules.html_generator import chat_html_wrapper
from modules.LoRA import add_lora_to_model
from modules.models import load_model, load_soft_prompt, unload_model,load_base_model
# from modules.text_generation import (generate_reply_wrapper,
#                                      get_encoded_length, stop_everything_event)


from extensions.api import script as api_script


if __name__ == "__main__":
    # Loading custom settings
    settings_file = None
    if shared.args.settings is not None and Path(shared.args.settings).exists():
        settings_file = Path(shared.args.settings)
    elif Path('settings.yaml').exists():
        settings_file = Path('settings.yaml')
    elif Path('settings.json').exists():
        settings_file = Path('settings.json')

    if settings_file is not None:
        logger.info(f"Loading settings from {settings_file}...")
        file_contents = open(settings_file, 'r', encoding='utf-8').read()
        new_settings = json.loads(file_contents) if settings_file.suffix == "json" else yaml.safe_load(file_contents)
        for item in new_settings:
            shared.settings[item] = new_settings[item]

  
    # Default extensions
    extensions_module.available_extensions = utils.get_available_extensions()
    if shared.is_chat():
        for extension in shared.settings['chat_default_extensions']:
            shared.args.extensions = shared.args.extensions or []
            if extension not in shared.args.extensions:
                shared.args.extensions.append(extension)
    else:
        for extension in shared.settings['default_extensions']:
            shared.args.extensions = shared.args.extensions or []
            if extension not in shared.args.extensions:
                shared.args.extensions.append(extension)

    available_models = utils.get_available_models()

    # Model defined through --model
    # if shared.args.model is not None:
    #     shared.model_name = shared.args.model

    # # Only one model is available
    # elif len(available_models) == 1:
    #     shared.model_name = available_models[0]

    # # Select the model from a command-line menu
    # elif shared.args.model_menu:
    #     if len(available_models) == 0:
    #         logger.error('No models are available! Please download at least one.')
    #         sys.exit(0)
    #     else:
    #         print('The following models are available:\n')
    #         for i, model in enumerate(available_models):
    #             print(f'{i+1}. {model}')

    #         print(f'\nWhich one do you want to load? 1-{len(available_models)}\n')
    #         i = int(input()) - 1
    #         print()

    #     shared.model_name = available_models[i]

    # If any model has been selected, load it
    # if shared.model_name != 'None':
    #     # model_settings = get_model_specific_settings(shared.model_name)
    #     shared.settings.update(model_settings)  # hijacking the interface defaults
    #     update_model_parameters(model_settings, initial=True)  # hijacking the command-line arguments

#zmy    
        # shared.model['base_model'], shared.tokenizer = load_base_model(shared.model_name)
        # for s in os.listdir('characters/'):
        # for cha in [shared.Harry_Potter,shared.Elon_Musk,shared.Sherlock,shared.Vladimir_Putin]:
            # s = f'{cha}'.split('.')[-1]
            # if s.split('.')[-1] == 'yaml':
                # s = s.split(".")[0]
                # print(f'loading {s}.....')
    api_script.setup()
    base_model, shared.tokenizer = load_base_model(shared.model_name)
    # delta_weight_path = yaml.safe_load(open('characters/'+s+'.yaml','r', encoding='utf-8').read())['delta_weight_path']
    
    shared.model = add_lora_to_model(base_model,shared.args.lora_weight_path)
    # base_model, shared.tokenizer = load_base_model(shared.model_name)
    # shared.tweet_model = add_lora_to_model(base_model,shared.args.tweet_model)
    print('tweet path:',shared.args.tweet_model)
    print('Successfully load lora!')

    # Force a character to be loaded
    if shared.is_chat():
        shared.persistent_interface_state.update({
            'mode': shared.settings['mode'],
            'character_dw_menu': shared.args.character or shared.settings['character'],
            'instruction_template': shared.settings['instruction_template']
        })

    shared.generation_lock = Lock()
    # Launch the web UI
    # create_interface()
    while True:
        time.sleep(0.5)
        if shared.need_restart:
            shared.need_restart = False
            time.sleep(0.5)
            shared.gradio['interface'].close()
            time.sleep(0.5)
            # create_interface()
    
