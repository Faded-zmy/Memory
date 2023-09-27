import json
import utils2
import re
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--split_id', type=str, default="0", help="split id")
args = parser.parse_args()

def get_response_from_gpt(prompt_path,question,conv_above):
    prompt = open(prompt_path,'r').read().replace('"Please Place the question there"',question).replace('"Please place conversation above the question there"',conv_above)
    # print('prompt:',prompt)
    # print('prompt_length',len(prompt))
    result = utils2.openai_chatcompletion2(prompt)
    return result

def transfer_context_to_json(context):
    splitted_data = re.split(f"(Subject|Atmosphere|Abstract|Opinion):", context)
    subject = splitted_data[2]
    atmosphere = splitted_data[4]
    abstract = splitted_data[6]
    opinion = splitted_data[8]
    result = {'Subject':subject,'Atmosphere':atmosphere,'Abstract':abstract,'Opinion':opinion}
    return result

def post_process(ids,question,result):
    splitted_data = re.split(f"(Context 1|Answer from Context 1|Context 2|Answer from Context 2|Comprehensive answer):", result)
    contexts = [transfer_context_to_json(splitted_data[2]),transfer_context_to_json(splitted_data[6])]
    answer = {'@1':splitted_data[4],'@2':splitted_data[8],'Comprehensive_answer':splitted_data[10]}
    if 'Context 3:' in result:
        context3 = transfer_context_to_json(result.split('Context 3:')[-1].split('Answer from Context 3:')[0].strip())
        contexts.append(context3)
        answer['@3']=result.split('Answer from Context 3:')[-1].split('Comprehensive_answer')[0]

    pp_result = {'id':ids,'question':question,'contexts':contexts,'answer':answer}
    return pp_result



if __name__ == '__main__':
    all_questions = json.load(open('/home/ec2-user/mengying/Data/memory/questions_from_60k_complete_{}.json'.format(args.split_id),'r'))
    prompt_path = '/home/ec2-user/mengying/Memory/generate_data/prompt_dao_context.txt'
    all_result = []
    for aq in tqdm(all_questions,total=len(all_questions)):
        try:
            ids = aq['id']
            text_above = aq['text_above']
            question = aq['question']
            result = get_response_from_gpt(prompt_path,question,text_above)
            # print('result:',result)
            pp_result = post_process(ids,question,result)
            # print('pp_result',pp_result)
            all_result.append(pp_result)
        except:
            pass
    f = open(f'./context_from_60k_complete_{args.split_id}.json','w')
    f.write(json.dumps(all_result,indent=3))
        


