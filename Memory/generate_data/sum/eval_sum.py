import utils
import openai
import json
import re
import time
from tqdm import tqdm
import tiktoken
from multiprocessing import Process



# Replace `your-api-key` with your actual API key
def openai_chatcompletion(prompt, api_id):
    api_key = [
    # "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",
    # "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",
    "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",
    # "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",
    ]
    openai.api_key = api_key[api_id]
    cn = 0
    enc = tiktoken.get_encoding("cl100k_base")
    # print(f"Tokens Of Prompt in {api_id}: {len(enc.encode(prompt))}!")
    if len(enc.encode(prompt)) > 7000:
        return None
    # max_token = 15000 - len(enc.encode(prompt))

    while True:
        output = ""
        try:
            response = openai.ChatCompletion.create(
                # model="gpt-3.5-turbo-16k",
                model="gpt-4",
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=1,
                # max_tokens=max_token,
                request_timeout = 120,
                timeout = 120
            )
            output = response['choices'][0]['message']['content']
            break
        except openai.error.OpenAIError as e:
            print(f"error {e} {api_id}")
            # logging.warning("Hit request rate limit; retrying...")
            #print("error cn", cn)
            # if e.json_body != None:
            #     if e.json_body['error']['type'] == 'invalid_request_error':
            #         max_token -= 500
            cn += 1
            if cn > 3:
                print(f"Already tried more than {cn} times!")
                break
            time.sleep(18)

    return output



def load_eval_prompt(Answer1, Answer2, conv):
    prompt = f"""
    We hope you can provide feedback on the performance of the two AI assistants to answer the user question displayed below. Assuming Answer1 is 90 points, based on the relevance of the answer to the entire conversation and the amount of information in the Answer2, give the Answer2 a score of 0-100 points. Please provide a comprehensive reason for your evaluation, avoid any potential biases, and ensure that the order of responses does not affect your judgment. The format is as follows:  \nScore: xxx
    
    Question: The following is a conversation between User_A and User_B. Please help to extract information and summarize opinion on both sides from the following conversation.
    
    The conversation is:
    [Start of conversation]
        {conv}
    [End of conversation]
    
    Answer1: 
    [Start of answer1]
        {Answer1}
    [End of answer1]

    Answer2:
    [Start of answer2]
        {Answer2}
    [End of answer2]

    """
    return prompt

def PostProcess(result):
    score_str=result.strip().lower().split('score: ')[1].split('\n')[0]
    # print("SCORE:", score_str)
    score=int(score_str)
    return score


def evaluate(Answer1, Answer2, conv):
    prompt = load_eval_prompt(Answer1, Answer2, conv)
    # print("PROMPT:", prompt)
    result = openai_chatcompletion(prompt, 0)
    # print("RESULT:", result)
    try: 
        score = PostProcess(result)
        return score
    except Exception as e:
        print(e)
        print("ERROR:", result)
    # return output



def work(start, end, data, split_id):
    score = 0
    num = 0
    for i in tqdm(range(start, end), total=end-start):
        gpt4_answer = data[i]['gpt4']
        our_answer = data[i]['our']
        conv = data[i]['conv']
        score_sg =  evaluate(gpt4_answer, our_answer, conv)
        if score_sg!=None:
            score+=score_sg
            num+=1
    print('-'*60)
    print("PROCESS:", split_id)
    print("SCORE:", score)
    print("NUM:", num)
    print('Average score:', score/num if num!=0 else 'None')
    print('-'*60)
    return score, num




def main():
    # with open('roomchat_person_selected_news.json', 'r', encoding='utf-8') as f:
    #     combinations = utils.jload(f)
    # print(combinations)
   
    # for root_path in folders:
        # root_path = '/ai_efs/lifeng/data/character/characters_dialogue/characters_dialogue_20230705_gpu2/'
        # combinations = [root_path+'/'+file for file in os.listdir(root_path) if file.endswith('.json')]
        data = json.load(open('/ai_jfs/mengying/data/sum/eval_data_llama2_half_0815.json', 'r'))
        num_process = 1
        division = len(data) // num_process
        print("EVAL NUM:", len(data))
        ranges = [i*division for i in range(num_process)]
        processes = []
        # output_dir = '/ai_jfs/mengying/data/sum/'+root_path.split('/')[-1]+'/'
        for i in range(num_process):
            # save_path = os.path.join(output_dir, "regen_roomchat_specific_news_"+str(i)+".json")
            if i == num_process - 1:
                start = ranges[i]
                end = len(data)
                p = Process(target=work, args=(start, end, data, i))
            else:
                start = ranges[i]
                end = ranges[i+1]
                p = Process(target=work, args=(start, end, data, i))
            p.start()
            processes.append(p)
            time.sleep(36)
        for p in processes:
            p.join()
 
if __name__ == '__main__':
    main()


# if __name__=="__main__":
#     score = 0
#     num = 0
    
#     # answer_sum = json.load(line)
#     question_ls = json.load(open('/ai_efs/mengying/FastChat/fastchat/eval/table/question_and_answer/get_result_question_zmy.jsonl', 'r'))
#     gpt4_answer = json.load(open('/home/ec2-user/mengying/Memory/generate_data/sum/sum_eval_gpt4_answer.jsonl', 'r'))
#     # for i in tqdm(range(len(answer_sum)), total = len(answer_sum)):
#     with open('/ai_efs/mengying/FastChat/fastchat/eval/table/question_and_answer/answer_sum.jsonl', 'r') as f:
#         for line in tqdm(f.readlines(), total=len(f.readlines())):
#             answer_sum = json.loads(line)
#             # print(answer_sum)
#             question = question_ls[int(answer_sum['question_id'])]
#             gpt4_res = gpt4_answer[int(answer_sum['question_id'])]
#             if answer_sum['question_id'] == question['id'] == gpt4_res['question_id']:
#                 conv = question['conversations'][0]['value'].split('The conversation is as follows:\n\n')[-1]
#                 Answer1 = gpt4_res['text']
#                 Answer2 = answer_sum['text']
#                 # print("="*60)
#                 # print("CONVERSAION:\n", conv)
#                 # print("="*60)
#                 # print("GPT4:\n", Answer1)
#                 # print("="*60)
#                 # print("SUM:\n", Answer2)
            
#                 print("="*60)
#                 while True:
#                     try:
#                         result = openai_chatcompletion(load_eval_prompt(Answer1, Answer2, conv), 0)
#                         break
#                     except Exception as e:
#                         print("GPT4 retry!")
#                         time.sleep(30)
                        
#                 print("="*60)
#                 try:
#                     score_str=result.strip().lower().split('score: ')[1].split('\n')[0]
#                     print("SCORE:", score_str)
#                     score+=int(score_str)
#                     num+=1
#                 except Exception as e:
#                     print(e)
#                     print(score_str)

# print("Conv Num:", num)        
# print("Average score", score/num)
            
            
            
    
# print("="*60)
# get_gpt_result(load_eval_prompt(Answer1, Answer2, conv))
# print("="*60)
               