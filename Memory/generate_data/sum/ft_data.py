import os
import json
import jsonlines
from tqdm import tqdm
def load_prompt(conv):
    prompt = """The following is a conversation between User_A and User_B. Please help to extract information and summarize opinion on both sides from the following conversation.
    Requirement:
    1. Divide the information into two parts, a description of people and opinions on a topic.
    2. Summarize the conversation into one topic and the opinion of both sides. Give both sides's  way of talking and did he achieve a achievement such as convincing someone, getting a message when discuss the topic using a concise sentence or some words.
    3. For the description of people, construct an information card of both sides.
    4. "Todo" is what people is going to do, "Todo_Time" is corresponding time.
    5. If a certain key's information is not mentioned, fill in it with "None".

    The structure of the information card is as follows:
    {"basic_information": {"Name": xxx, "Gender": xxx, "Date_of_Birth": xxx, "Place_of_Birth": xxx, "Race": xxx, "Nationality": xxx}, "background": {"Educational_Background": xxx, "Occupation": xxx, "Position": xxx, "Achievement": xxx}, "others": {"Personality": xxx, "Hobbies": xxx, "Good_At": xxx, "Not_Good_At": xxx, "Topics_of_Interest": xxx, "Topics_of_Disinterest": xxx, "People_They_Admire": xxx, "People_They_Dislike": xxx, "Todo": xxx, "Todo_Time": xxx}}

    The output format is as follows:
    User_A's information card: xxx

    User_B's information card: xxx

    Discussions:
    {"Topic": xxx,
    "Summary": xxx,
    "User_A's_Opinion": xxx,
    "User_A's_Way_of_Talking": xxx,
    "User_A's_Achievement":xxx,
    "User_B's_Opinion": xxx,
    "User_B's_Way_of_Talking": xxx,
    "User_B's_Achievement":xxx}

    The conversation is as follows:\n
    """
    prompt = prompt+conv
    return prompt


def input_format(inputs, gt):
    res_one = {
                'id': "identity_{}".format(str(idt)),
                'conversations': [
                    {
                        "from": "human",
                        "value": inputs
                    },
                    {
                        "from": "gpt",
                        "value": gt
                    }
                ]
                }
    return res_one

def dic2dic(dic, category):
    key_map = {
        "bi": "basic_information",
        "bg": "background",
        "o": "others",
        "N": "Name",
        "Gd": "Gender",
        "DB": "Date_of_Birth",
        "PB": "Place_of_Birth",
        "R": "Race",
        "Nat": "Nationality",
        "EB": "Educational_Background",
        "Op": "Occupation",
        "Pos": "Position",
        "A": "Achievement",
        "Per": "Personality",
        "H": "Hobbies",
        "G": "Good_At",
        "Ng": "Not_Good_At",
        "Toi": "Topics_of_Interest",
        "Tod": "Topics_of_Disinterest",
        "PTA": "People_They_Admire",
        "PTD": "People_They_Dislike",
        "Td": "Todo",
        "TT": "Todo_Time",
        "T": "Topic",
        "Ao": "User_A's_Opinion",
        "Bo": "User_B's_Opinion",
        "Aw": "User_A's_Way_of_Talking",
        "Bbao bw": "User_B's_Way_of_Talking",
        "sum": "Summary",
        "Aa": "User_A's_Achievement",
        "Ba": "User_B's_Achievement"
    }
    trans_dic = {}
    if category == "information_card":
        for k in dic.keys():
            sub_dic = {}
            for sk in dic[k].keys():
                sub_dic[key_map[sk]] = dic[k][sk]
            trans_dic[key_map[k]] = sub_dic
    
    elif category == "discussion":

        for k in dic.keys():
            trans_dic[key_map[k]] = dic[k]
    return trans_dic

# # train data
# data_path = '/home/ec2-user/mengying/Memory/generate_data/sum/data/'
# root_path = [data_path+p for p in os.listdir(data_path)]
# root_path.append('/ai_efs/mengying/data/sum/characters_dialogue_20230614_gpu1')
# res = []
# idt = 0
# for rp in tqdm(root_path, total=len(root_path)):
#     for json_name in tqdm(os.listdir(rp), total=len(os.listdir(rp))):
#         data_org = json.load(open(rp+'/'+json_name, 'r'))
#         for i in range(len(data_org['org_conv'])):
#             inputs = load_prompt(data_org['org_conv'][i])
#             A_info = dic2dic(data_org['A_info'][i], 'information_card')
#             B_info = dic2dic(data_org['B_info'][i], 'information_card')
#             # print("DISCUSSION:", data_org['Dis'][i])
#             Dis = dic2dic(data_org['Dis'][i], 'discussion')
#             # print("DIS:", str(Dis))
#             gt = "User_A's information card:\n"+str(A_info)+"\n\nUser_B's information card:\n"+str(B_info)+"\n\nDiscussions:\n"+str(Dis)
#             # print("GT:", gt)
#             res.append(input_format(inputs, gt))
#             idt+=1
# json.dump(res, open('/ai_efs/mengying/data/sum/train_data_0724.json', 'w'), indent=3)

# eval data
root_path = '/ai_efs/mengying/data/sum/characters_dialogue_20230616_gpu2/'
json_path = os.listdir(root_path)
idx = 0
question_file, question_get_result, answer_gpt4 = [], [], []
# question_file = jsonlines.open('./sum_eval_question.jsonl', 'w')
# question_get_result = jsonlines.open('./get_result_question.jsonl', 'w')
# answer_gpt4_file = jsonlines.open('./sum_eval_gpt4_answer.jsonl', 'w')
# answer_our = []
for jp in json_path:
    data = json.load(open(root_path+jp, 'r'))
    for i in range(len(data['org_conv'])):
        # question
        question = load_prompt(data['org_conv'][i])
        question_file.append({'question_id': str(idx), 'text': question, 'category': 'sum'})
        # jsonlines.Writer.write(question_file, {'question_id': str(idx), 'text': question, 'category': 'sum'})

        # gpt4 result
        A_info = dic2dic(data['A_info'][i], 'information_card')
        B_info = dic2dic(data['B_info'][i], 'information_card')
        Dis = dic2dic(data['Dis'][i], 'discussion')
        gt = "User_A's information card:\n"+str(A_info)+"\n\nUser_B's information card:\n"+str(B_info)+"\n\nDiscussions:\n"+str(Dis)
        answer_gpt4.append({'question_id': str(idx), 'text': gt})
        # jsonlines.Writer.write(answer_gpt4_file, {'question_id': str(idx), 'text': gt})

        # question to get our result
        question_get_result.append({'id': str(idx), 'conversations': [{'from': 'human', 'value':question}]})
        # jsonlines.Writer.write(question_get_result, {'id': str(idx), 'conversations': [{'from': 'human', 'value':question}]})

        idx+=1
        # our result 等待生成

json.dump(question_file, open('./sum_eval_question.jsonl', 'w'))
json.dump(question_get_result, open('./get_result_question.jsonl', 'w'))
json.dump(answer_gpt4, open('./sum_eval_gpt4_answer.jsonl', 'w'))