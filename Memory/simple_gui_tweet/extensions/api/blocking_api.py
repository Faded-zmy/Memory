import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

from extensions.api.util import build_parameters, try_start_cloudflared
from modules import shared
# from modules.chat import generate_chat_reply,load_character
from modules.text_generation import generate_chat_reply,generate_tweet #encode, generate_reply,



class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/api/v1/model':
            self.send_response(200)
            self.end_headers()
            response = json.dumps({
                'result': shared.model_name
            })

            self.wfile.write(response.encode('utf-8'))
        else:
            self.send_error(404)

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        body = json.loads(self.rfile.read(content_length).decode('utf-8'))

        if self.path == '/api/v1/generate':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()

            prompt = body['prompt']
            generate_params = build_parameters(body)
            stopping_strings = generate_params.pop('stopping_strings')
            generate_params['stream'] = False
            shared.mem_root = body['mem_root']
            prompt = body['context']+'\n'+prompt
            generator = generate_reply(
                prompt, generate_params, stopping_strings=stopping_strings, is_chat=False)

            answer = ''
            for a in generator:
                answer = a

            response = json.dumps({
                'results': [{
                    'text': answer
                }]
            })

            self.wfile.write(response.encode('utf-8'))

        elif self.path == '/api/v1/chat':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            regenerate = body.get('regenerate', False)
            _continue = body.get('_continue', False)
            generate_params = build_parameters(body,chat=True)
            stopping_strings = generate_params.pop('stopping_strings')
            generate_params['stream'] = False
            # shared.mem_root = body['mem_root']
            # prompt = body['context']+'\n'+prompt
            if body['mode'] == 'chat':
                prompt = generate_params['context']
                for mes in body['messages']:
                    if mes['role'].lower() == 'user':
                        # prompt = prompt+'\nYou: '+mes['content']
                        prompt = prompt+'\You: '+mes['content']

                    elif mes['role'].lower() == 'assistant':
                        prompt = prompt+'\nAssistant: '+mes['content']+'</s>'
                history = []#TODO history传递可优化
                for i in range(len(body['messages'])//2):
                    history.append([body['messages'][i*2]['content'],body['messages'][i*2+1]['content']])
                prompt = prompt+'\nAssistant: '

                generator = generate_chat_reply(
                    generate_params['name1'], generate_params['name2'], history, prompt)
            elif body['mode'] == 'tweet':
                generator = generate_tweet(
                    body['news'],body['tweet_style']
                )
            # answer = history
            # for a in generator:
            #     answer = a
            response = json.dumps({
                'results': [{
                    'history': generator
                }]
            })

            self.wfile.write(response.encode('utf-8'))

        elif self.path == '/api/v1/token-count':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()

            tokens = encode(body['prompt'])[0]
            response = json.dumps({
                'results': [{
                    'tokens': len(tokens)
                }]
            })

            self.wfile.write(response.encode('utf-8'))
        else:
            self.send_error(404)


def _run_server(port: int, share: bool = False):
    address = '0.0.0.0' if shared.args.listen else '127.0.0.1'
    print('ZMY-port',address,port)
    server = ThreadingHTTPServer((address, port), Handler)

    def on_start(public_url: str):
        print(f'Starting non-streaming server at public url {public_url}/api')


    if share:
        try:
            try_start_cloudflared(port, max_attempts=3, on_start=on_start)
        except Exception:
            pass
    else:
        print(
            f'Starting API at http://{address}:{port}/api')

    server.serve_forever()


def start_server(port: int, share: bool = False):
    Thread(target=_run_server, args=[port, share], daemon=True).start()
