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
# from modules.html_generator import generate_cai_chat_html
from modules.LoRA import add_lora_to_model
from modules.models import load_model, load_soft_prompt, unload_model,load_base_model



def create_interface():

    # Defining some variables
    gen_events = []
    title = 'Text generation web UI'

    # Importing the extension files and executing their setup() functions
    # if shared.args.extensions is not None and len(shared.args.extensions) > 0:
    #     extensions_module.load_extensions()

    # css/js strings
    css = ui.css if not shared.is_chat() else ui.css + ui.chat_css
    js = ui.main_js if not shared.is_chat() else ui.main_js + ui.chat_js
    css += apply_extensions('css')
    js += apply_extensions('js')

    with gr.Blocks(css=css, analytics_enabled=False, title=title, theme=ui.theme) as shared.gradio['interface']:
        if Path("notification.mp3").exists():
            shared.gradio['audio_notification'] = gr.Audio(interactive=False, value="notification.mp3", elem_id="audio_notification", visible=False)
            audio_notification_js = "document.querySelector('#audio_notification audio')?.play();"
        else:
            audio_notification_js = ""

        # Create chat mode interface
        if shared.is_chat():
            shared.input_elements = ui.list_interface_input_elements(chat=True)
            shared.gradio['interface_state'] = gr.State({k: None for k in shared.input_elements})
            shared.gradio['Chat input'] = gr.State()
            # shared.gradio['dummy'] = gr.State()

            with gr.Tab('Text generation', elem_id='main'):
                # shared.gradio['display'] = gr.HTML(value=chat.generate_cai_chat_html(),lines=15)
                # shared.gradio['textbox'] = gr.Textbox(label='Input',scale=1 ,min_width=100)
                with gr.Column():
                    with gr.Row():
                        with gr.Column(scale=0.5):
                            # with gr.Box(scale=0.5):
                            # with gr.Column(scale=0.5):
                            #     gr.Label(value= 'Please choose character!',color= 'blue')
                            with gr.Column(scale=0.5):
                                shared.gradio['character_menu'] = gr.Dropdown(choices=utils.get_available_characters(), value = shared.character,label='Character', elem_id='character-menu', info='Used in chat and tweet modes.')
                        with gr.Row(scale=8):
                            shared.gradio['display'] = gr.HTML(value=chat.generate_cai_chat_html(),lines=15)
                        # with gr.Row(scale=3):
                        #     # with gr.Box():
                        #     with gr.Row():
                        #         gr.Label(value= 'Please choose character!',color= 'blue')
                        #     with gr.Row():
                        #         shared.gradio['character_menu'] = gr.Dropdown(choices=utils.get_available_characters(), value = shared.character,label='Character', elem_id='character-menu', info='Used in chat and chat-instruct modes.')
                    with gr.Column(scale=8):
                        with gr.Column():
                            shared.gradio['textbox'] = gr.Textbox(label='Input',scale=1 ,min_width=100)

                        with gr.Column():
                            shared.gradio['Generate'] = gr.Button('Generate', elem_id='Generate', variant='primary')

                        with gr.Column():
                            # shared.gradio['Remove last'] = gr.Button('Remove last')
                            shared.gradio['Clear history'] = gr.Button('Clear history')
                            shared.gradio['Clear history-confirm'] = gr.Button('Confirm', variant='stop', visible=False)
                            shared.gradio['Clear history-cancel'] = gr.Button('Cancel', visible=False)
                    
                

            #zmy TODO
        # if shared.args.tweet_model != None:
            with gr.Tab('Tweet generation', elem_id='main'):
                with gr.Box():
                    with gr.Row():
                        with gr.Column(scale=4):
                            shared.gradio['news'] = gr.Textbox(label='news',lines=27)

                        with gr.Column(scale=8):
                            with gr.Row():
                                gr.Label(value= 'news_display')
                            with gr.Row():
                                shared.gradio['news_display'] = gr.HTML(label='input news',value=shared.news,show_label=True)
                        with gr.Column(scale=8):
                            with gr.Row():
                                gr.Label(value= 'tweet')
                            with gr.Row():
                                shared.gradio['tweet_display'] = gr.HTML(label='tweet',value=shared.tweet,show_label = True)
                    with gr.Row():
                        shared.gradio['tweet_style'] = gr.Radio(choices=['humorous',  'serious'],label='Style', info='the tweet style')
                    with gr.Row():
                        shared.gradio['tweet_generate'] = gr.Button('tweet_generate')




            
            with gr.Tab('Chat settings', elem_id='chat-settings'):
                with gr.Row():
                    with gr.Column(scale=8):
                        with gr.Row():
                            # shared.gradio['character_menu'] = gr.Dropdown(choices=utils.get_available_characters(), label='Character', elem_id='character-menu', info='Used in chat and chat-instruct modes.')
                            # ui.create_refresh_button(shared.gradio['character_menu'], lambda: None, lambda: {'choices': utils.get_available_characters()}, 'refresh-button')
                            shared.gradio['save_character'] = ui.create_save_button(elem_id='refresh-button')
                            shared.gradio['delete_character'] = ui.create_delete_button(elem_id='refresh-button')

                        shared.gradio['save_character-filename'] = gr.Textbox(lines=1, label='File name:', interactive=True, visible=False)
                        shared.gradio['save_character-confirm'] = gr.Button('Confirm save character', elem_classes="small-button", variant='primary', visible=False)
                        shared.gradio['save_character-cancel'] = gr.Button('Cancel', elem_classes="small-button", visible=False)
                        shared.gradio['delete_character-confirm'] = gr.Button('Confirm delete character', elem_classes="small-button", variant='stop', visible=False)
                        shared.gradio['delete_character-cancel'] = gr.Button('Cancel', elem_classes="small-button", visible=False)

                        shared.gradio['name1'] = gr.Textbox(value=shared.settings['name1'], lines=1, label='Your name')
                        shared.gradio['name2'] = gr.Textbox(value=shared.settings['name2'], lines=1, label='Character\'s name')
                        shared.gradio['context'] = gr.Textbox(value=shared.settings['context'], lines=4, label='Context')
                        shared.gradio['greeting'] = gr.Textbox(value=shared.settings['greeting'], lines=4, label='Greeting')
                        # shared.gradio['delta_weight_path'] = gr.Textbox(value=shared.settings['delta_weight_path'], lines=1, label='delta_weight_path')

                    with gr.Column(scale=1):
                        shared.gradio['character_picture'] = gr.Image(label='Character picture', type='pil')
                        shared.gradio['your_picture'] = gr.Image(label='Your picture', type='pil', value=Image.open(Path('cache/pfp_me.png')) if Path('cache/pfp_me.png').exists() else None)

                with gr.Row():
                    shared.gradio['instruction_template'] = gr.Dropdown(choices=utils.get_available_instruction_templates(), label='Instruction template', value='None', info='Change this according to the model/LoRA that you are using. Used in instruct and chat-instruct modes.')
                    ui.create_refresh_button(shared.gradio['instruction_template'], lambda: None, lambda: {'choices': utils.get_available_instruction_templates()}, 'refresh-button')

                shared.gradio['name1_instruct'] = gr.Textbox(value='', lines=2, label='User string')
                shared.gradio['name2_instruct'] = gr.Textbox(value='', lines=1, label='Bot string')
                shared.gradio['context_instruct'] = gr.Textbox(value='', lines=4, label='Context')
                shared.gradio['turn_template'] = gr.Textbox(value=shared.settings['turn_template'], lines=1, label='Turn template', info='Used to precisely define the placement of spaces and new line characters in instruction prompts.')
                with gr.Row():
                    shared.gradio['chat-instruct_command'] = gr.Textbox(value=shared.settings['chat-instruct_command'], lines=4, label='Command for chat-instruct mode', info='<|character|> gets replaced by the bot name, and <|prompt|> gets replaced by the regular chat prompt.')

                with gr.Row():
                    with gr.Tab('Chat history'):
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown('### Upload')
                                shared.gradio['upload_chat_history'] = gr.File(type='binary', file_types=['.json', '.txt'])

                            with gr.Column():
                                gr.Markdown('### Download')
                                shared.gradio['download'] = gr.File()
                                shared.gradio['download_button'] = gr.Button(value='Click me')

               
                        gr.Markdown('### TavernAI PNG format')
                        shared.gradio['upload_img_tavern'] = gr.File(type='binary', file_types=['image'])

          

        
        # chat mode event handlers
        if shared.is_chat():
            
            clear_arr = [shared.gradio[k] for k in ['Clear history-confirm', 'Clear history', 'Clear history-cancel']]
            # shared.reload_inputs = [shared.gradio[k] for k in ['name1', 'name2', 'chat_style']]

            # gen_events.append(shared.gradio['Generate'].click(
            #     ui.gather_interface_values, [shared.gradio[k] for k in shared.input_elements], shared.gradio['interface_state']).then(
            #     lambda x: (x, ''), shared.gradio['textbox'], [shared.gradio['Chat input'], shared.gradio['textbox']], show_progress=False).then(
            #     chat.generate_chat_reply_wrapper, shared.gradio['Chat input'], shared.gradio['display'], show_progress=False).then(
            #     chat.save_history, 'chat', None, show_progress=False).then(
            #     lambda: None, None, None, _js=f"() => {{{audio_notification_js}}}")
            # )
            shared.gradio['max_new_tokens'] = 2048
            # shared.gradio['URI'] = []
            gen_events.append(shared.gradio['Generate'].click(
                ui.gather_interface_values, [shared.gradio[k] for k in shared.input_elements], shared.gradio['interface_state']).then(
                lambda x: (x, ''), shared.gradio['textbox'], [shared.gradio['Chat input'], shared.gradio['textbox']], show_progress=False).then(
                # lambda x: (x, ''),shared.gradio['character_menu'], shared.cha_name).then(
                chat.get_api_chat_result, [shared.gradio['character_menu'],shared.gradio['Chat input']], None, show_progress=False).then(
                chat.save_history, shared.gradio['Chat input'], None, show_progress=False).then(
                chat.generate_cai_chat_html, None, shared.gradio['display']).then(
                lambda: None, None, None, _js=f"() => {{{audio_notification_js}}}")
            )
            
            
            #zmy TODO
          
            gen_events.append(shared.gradio['tweet_generate'].click(
                lambda x: (x, ''), shared.gradio['news'], [shared.gradio['news_display'], shared.gradio['news']], show_progress=False).then(
                lambda y: (y, ''), shared.gradio['tweet_style'], shared.gradio['tweet_style'],show_progress=False).then(
                chat.get_api_tweet_result, [shared.gradio['character_menu'],shared.gradio['news'],shared.gradio['tweet_style']],shared.gradio['tweet_display'], show_progress=False).then(
                lambda: None, None, None, _js=f"() => {{{audio_notification_js}}}")
            )

      

            # gen_events.append(shared.gradio['textbox'].submit(
            #     ui.gather_interface_values, [shared.gradio[k] for k in shared.input_elements], shared.gradio['interface_state']).then(
            #     lambda x: (x, ''), shared.gradio['textbox'], [shared.gradio['Chat input'], shared.gradio['textbox']], show_progress=False).then(
            #     chat.generate_chat_reply_wrapper, shared.input_params, shared.gradio['display'], show_progress=False).then(
            #     chat.save_history,'chat', None, show_progress=False).then(
            #     lambda: None, None, None, _js=f"() => {{{audio_notification_js}}}")
            # )

            shared.gradio['Clear history-confirm'].click(
                lambda: [gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)], None, clear_arr).then(
                chat.clear_chat_log, None, None).then(
                # chat.save_history, 'chat', None, show_progress=False).then(
                chat.redraw_html, shared.reload_inputs, shared.gradio['display'])  
            # shared.gradio['Clear history-confirm'].click(
            #     lambda: [gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)], None, clear_arr).then(
            #     chat.clear_chat_log, [shared.gradio[k] for k in ['greeting']], None).then(
            #     chat.save_history, 'chat', None, show_progress=False).then(
            #     chat.redraw_html, shared.reload_inputs, shared.gradio['display'])

         
            shared.gradio['Clear history'].click(lambda: [gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)], None, clear_arr)
            shared.gradio['Clear history-cancel'].click(lambda: [gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)], None, clear_arr)
           
            # shared.gradio['character_menu'].change(
            #     partial(chat.load_character, instruct=False), [shared.gradio[k] for k in ['character_menu', 'name1', 'name2']], [shared.gradio[k] for k in ['name1', 'name2', 'character_picture', 'greeting', 'context']]).then(
            #     chat.redraw_html, shared.reload_inputs, shared.gradio['display'])

            shared.gradio['character_menu'].change(
                partial(chat.load_character, instruct=False), shared.gradio['character_menu'], shared.gradio['character_picture']).then(
                chat.redraw_html, shared.reload_inputs, shared.gradio['display'])
       
            shared.gradio['your_picture'].change(
                chat.upload_your_profile_picture, shared.gradio['your_picture'], None).then(
                partial(chat.redraw_html, reset_cache=True), shared.reload_inputs, shared.gradio['display'])

      
        extensions_module.create_extensions_block()
    shared.gradio['interface'].queue()
    auth = None
    if shared.args.listen:
        shared.gradio['interface'].launch(prevent_thread_lock=True, share=shared.args.share, server_name=shared.args.listen_host or '0.0.0.0', server_port=shared.args.listen_port, inbrowser=shared.args.auto_launch, auth=auth)
    else:
        shared.gradio['interface'].launch(prevent_thread_lock=True, share=shared.args.share, server_port=shared.args.listen_port, inbrowser=shared.args.auto_launch, auth=auth)


   


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
    if shared.args.model is not None:
        shared.model_name = shared.args.model

    # Only one model is available
    elif len(available_models) == 1:
        shared.model_name = available_models[0]

    # Select the model from a command-line menu
    elif shared.args.model_menu:
        if len(available_models) == 0:
            logger.error('No models are available! Please download at least one.')
            sys.exit(0)
        else:
            print('The following models are available:\n')
            for i, model in enumerate(available_models):
                print(f'{i+1}. {model}')

            print(f'\nWhich one do you want to load? 1-{len(available_models)}\n')
            i = int(input()) - 1
            print()

        shared.model_name = available_models[i]

    # If any model has been selected, load it
    if shared.model_name != 'None':
        model_settings = get_model_specific_settings(shared.model_name)
        shared.settings.update(model_settings)  # hijacking the interface defaults
        update_model_parameters(model_settings, initial=True)  # hijacking the command-line arguments


    # Force a character to be loaded
    if shared.is_chat():
        shared.persistent_interface_state.update({
            'mode': shared.settings['mode'],
            'character_dw_menu': shared.args.character or shared.settings['character'],
            'instruction_template': shared.settings['instruction_template']
        })

    shared.generation_lock = Lock()
    # Launch the web UI
    create_interface()
    while True:
        time.sleep(0.5)
        if shared.need_restart:
            shared.need_restart = False
            time.sleep(0.5)
            shared.gradio['interface'].close()
            time.sleep(0.5)
            create_interface()
