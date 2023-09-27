

import re
import os
import json
# print("+"*60)
# print("原来对话")
# print(conv)
# print("+"*60)
# final_conv = re.sub(r'\n.*Inner thoughts:.*', "", conv, flags=re.I)
# final_conv = re.sub(r'\n.*actions/expressions: .*', "", final_conv, flags=re.I)
# final_conv = re.sub('\n.*answer:', "", final_conv, flags=re.I)
# final_conv = re.sub('Person', "A", final_conv, flags=re.I)
# final_conv = re.sub('Character', "B", final_conv, flags=re.I)
# print("删了之后")
# print(final_conv)

# def filter_character(file_name):

#    cha_ls = ['Harry_Potter', 'Sherlock_Holmes', 'Arya_Stark', 'Sansa_Stark', 'Hermione_Granger', 'Ron_Weasley', 'Superman', 'Batman', 'Spider', 'Iron_Man', 'Captain_America', 'Wonder_Woman', 'Thor', 'Romeo', 'Juliet', 'Dr._Watson', 'Homer_Simpson', 'Bart_Simpson', 'SpongeBob_SquarePants', 'Patrick_Star', 'Pikachu', 'Ash_Ketchum', 'Lara_Croft', 'Hamlet', 'Jay_Gatsby', 'Elon_Musk', 'Steve_Jobs', 'Bill_Gates', 'Socrates', 'Albert_Einstein', 'Nikola_Tesla', 'Leonardo_da_Vinci', 'Charles_Darwin', 'Mark_Zuckerberg', 'Warren_Buffett', 'Martin_Luther_King_Jr.', 'Barack_Obama', 'J.K._Rowling', 'William_Shakespeare', 'Vincent_van_Gogh', 'Ludwig_van_Beethoven', 'Michael_Jackson', 'Lionel_Messi', 'Michael_Jordan', 'Mike_Tyson', 'Tiger_Woods', 'Arnold_Schwarzenegger', 'Jackie_Chan', 'Robert_Downey_Jr.', 'Leonardo_DiCaprio', 'Neil_Armstrong', 'Stephen_Hawking', 'Chow_Yun', 'Jay_Chou', 'Zhang_Yimou', 'Lin_Dan', 'Princess_Diana', 'Winston_Churchill', 'Marie_Curie', 'Donald_Trump']
#    for cha in cha_ls:

#       if cha in file_name:
#          return True 
#    return False

# file_name = "/ai_efs/lifeng/data/character/characters_dialogue/characters_dialogue_20230616_gpu2/Sherlock_Holmes_fore_1.json"
# print(filter_character(file_name))

# import os
# result = []
# root = "/ai_efs/lifeng/data/character/characters_dialogue/"
# for path in os.listdir("/ai_efs/lifeng/data/character/characters_dialogue/"):
#    result.append(root+path)
# print(result[10:])
# print(len(result))

# ['/ai_efs/lifeng/data/character/characters_dialogue/characters_dialogue_20230616_gpu1', '/ai_efs/lifeng/data/character/characters_dialogue/characters_dialogue_20230705_gpu2', '/ai_efs/lifeng/data/character/characters_dialogue/characters_dialogue_20230621_gpu1', '/ai_efs/lifeng/data/character/characters_dialogue/characters_dialogue_20230619_gpu2', '/ai_efs/lifeng/data/character/characters_dialogue/characters_dialogue_20230628_gpu2', '/ai_efs/lifeng/data/character/characters_dialogue/characters_dialogue_20230608_gpu2', '/ai_efs/lifeng/data/character/characters_dialogue/characters_dialogue_20230613_gpu2', '/ai_efs/lifeng/data/character/characters_dialogue/characters_dialogue_20230614_gpu1', '/ai_efs/lifeng/data/character/characters_dialogue/characters_dialogue_20230616_gpu2', '/ai_efs/lifeng/data/character/characters_dialogue/characters_dialogue_20230608_gpu1', '/ai_efs/lifeng/data/character/characters_dialogue/characters_dialogue_20230621_gpu2', '/ai_efs/lifeng/data/character/characters_dialogue/characters_dialogue_20230613_gpu1', '/ai_efs/lifeng/data/character/characters_dialogue/characters_dialogue_related_interest_20230614_gpu1', '/ai_efs/lifeng/data/character/characters_dialogue/characters_dialogue_20230619_gpu1', '/ai_efs/lifeng/data/character/characters_dialogue/characters_dialogue_20230628_gpu1', '/ai_efs/lifeng/data/character/characters_dialogue/characters_dialogue_20230614_gpu2', '/ai_efs/lifeng/data/character/characters_dialogue/characters_dialogue_20230705_gpu1']
   
# import json
# text = """{"basic_information": {"Name": 'xxx', "Gender": 'xxx', "Date_of_Birth": 'xxx', "Place_of_Birth": 'xxx', "Race": 'xxx', "Nationality": 'xxx'}}"""
# json.dump('./scratch.json',json.loads(text))
conv = """
"""
def PreProcess(org_conv):
    final_conv = re.sub(r'Inner thoughts:.*', "", org_conv, flags=re.I)
    final_conv = re.sub(r'\n.*actions/expressions:.*', "", final_conv, flags=re.I)
    final_conv = re.sub('\n.*answer:', "", final_conv, flags=re.I)
    # final_conv = re.sub('Person', "User_A", final_conv, flags=re.I)
    # final_conv = re.sub('Character', "User_B", final_conv, flags=re.I)
    return final_conv

res = []
lf_data = json.load(open('/ai_efs/lifeng/data/character/train_val_data/combine_train_data/2023-07-25_conbine_train_data_19_people_gpt4.json', 'r'))
for lfd in lf_data:
    conversation = ''
    for conv in lfd['conversations']:
        if conv['from'] == 'human':
            cha = 'User_A'
        else:
            cha = 'User_B'
        conversation = conversation+cha+": "+conv['value']+'\n'
        conversation = PreProcess(conversation)
    res.append({'id':lfd['id'], 'conv':conversation})
json.dump(res, open('/ai_efs/mengying/data/sum/lf_gpt4_19_data.json', 'w'), indent=4)
# print(PreProcess(conv))
# root_path = "/ai_efs/mengying/data/sum/characters_dialogue_20230614_gpu1/"
# jsons = os.listdir(root_path)
# result = {'discussion':[]}
# for j in jsons:
#     if "debate" in j:
#         dis = json.load(open(root_path+j, 'r'))['Dis']
#         result['discussion'].extend(dis)
# json.dump(result, open('./debate.json', 'w'), indent=4)
# res = 0
# root_path = '/ai_efs/mengying/data/sum/characters_dialogue_20230616_gpu2/'
# for rp in os.listdir(root_path):
#     res+=len(json.load(open(root_path+rp, 'r')))
# print(res)



