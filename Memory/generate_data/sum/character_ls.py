cha_ls_path1 = "/home/ec2-user/mengying/Memory/generate_data/sum/fiction_character_list_25.csv"
cha_ls_path2 = "/ai_efs/lifeng/data/character/name_list/real_character_list_25.csv"
cha_ls = open(cha_ls_path1,'r').readlines()[1:] + open(cha_ls_path2,'r').readlines()[1:]
result = []
for cha in cha_ls:
    cha = cha.split('-')[0].strip().replace(' ','_').replace('"','')
    result.append(cha)
print(result)
print(len(result))