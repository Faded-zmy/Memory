import os
import re
from pathlib import Path

from modules import shared


def atoi(text):
    return int(text) if text.isdigit() else text.lower()


# Replace multiple string pairs in a string
def replace_all(text, dic):
    for i, j in dic.items():
        text = text.replace(i, j)

    return text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def get_available_models():
    if shared.args.flexgen:
        return sorted([re.sub('-np$', '', item.name) for item in list(Path(f'{shared.args.model_dir}/').glob('*')) if item.name.endswith('-np')], key=natural_keys)
    else:
        return sorted([re.sub('.pth$', '', item.name) for item in list(Path(f'{shared.args.model_dir}/').glob('*')) if not item.name.endswith(('.txt', '-np', '.pt', '.json', '.yaml'))], key=natural_keys)


def get_available_presets():
    return sorted(set((k.stem for k in Path('presets').glob('*.yaml'))), key=natural_keys)


def get_available_prompts():
    prompts = []
    files = set((k.stem for k in Path('prompts').glob('*.txt')))
    prompts += sorted([k for k in files if re.match('^[0-9]', k)], key=natural_keys, reverse=True)
    prompts += sorted([k for k in files if re.match('^[^0-9]', k)], key=natural_keys)
    prompts += ['Instruct-' + k for k in get_available_instruction_templates() if k != 'None']
    prompts += ['None']
    return prompts


def get_available_characters(): #ZMY TODO
    paths = (x for x in Path('characters').iterdir() if x.suffix in ('.json', '.yaml', '.yml'))
    return ['None'] + sorted(set((k.stem for k in paths if k.stem != "instruction-following")), key=natural_keys)

def get_available_characters_delta_weight(): #ZMY TODO
    paths = os.listdir('/ai_efs/models/')
    return paths
    # paths = (x for x in Path('characters_delta_weight').iterdir() if x.suffix in ('.json', '.yaml', '.yml'))
    # return ['None'] + sorted(set((k.stem for k in paths if k.stem != "instruction-following")), key=natural_keys)

def get_available_instruction_templates():
    path = "characters/instruction-following"
    paths = []
    if os.path.exists(path):
        paths = (x for x in Path(path).iterdir() if x.suffix in ('.json', '.yaml', '.yml'))

    return ['None'] + sorted(set((k.stem for k in paths)), key=natural_keys)


def get_available_extensions():
    return sorted(set(map(lambda x: x.parts[1], Path('extensions').glob('*/script.py'))), key=natural_keys)


def get_available_softprompts():
    return ['None'] + sorted(set((k.stem for k in Path('softprompts').glob('*.zip'))), key=natural_keys)


def get_available_loras():
    return sorted([item.name for item in list(Path(shared.args.lora_dir).glob('*')) if not item.name.endswith(('.txt', '-np', '.pt', '.json'))], key=natural_keys)


def get_datasets(path: str, ext: str):
    return ['None'] + sorted(set([k.stem for k in Path(path).glob(f'*.{ext}') if k.stem != 'put-trainer-datasets-here']), key=natural_keys)


def get_available_chat_styles():
    return sorted(set(('-'.join(k.stem.split('-')[1:]) for k in Path('css').glob('chat_style*.css'))), key=natural_keys)
