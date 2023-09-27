import utils
import openai
import json
import re
import time
import tiktoken
def PreProcess(org_conv):
    final_conv = re.sub(r'\n.*Inner thoughts:.*', "", org_conv, flags=re.I)
    final_conv = re.sub(r'\n.*actions/expressions:.*', "", final_conv, flags=re.I)
    final_conv = re.sub('\n.*answer:', "", final_conv, flags=re.I)
    final_conv = re.sub('Person', "User_A", final_conv, flags=re.I)
    final_conv = re.sub('Character', "User_B", final_conv, flags=re.I)
    return final_conv

def PostProcess_IC(response):
    #转成dic
    splitted_data = re.split(f"(User_A's information card|User_B's.*information card|Discussions):", response, flags=re.I)
    A_info_card = splitted_data[2].strip().replace(': None',': "None"')
#     A_info_card = """
#     {"basic_information":'a'}
# """
#      A_info_card = """
#     {
# "basic_information": {"Name": User_A, "Gender": "None", "Date_of_Birth": "None", "Place_of_Birth": "None", "Race": "None", "Nationality": "None"}, 
# "background": {"Educational_Background": "None", "Occupation": "None", "Position": "None", "Achievement": "None"}, 
# "others": {"Personality": "Curious and supportive", "Hobbies": "None", "Good_at": "Talking and listening", "Not_good_at": "None", "Topics_of_interest": "User_B's life, relationship with father, fashion, and basketball", "Topics_of_disinterest": "None", "People_They_Admire": "None", "People_They_Dislike": "None", "Todo": "None", "Todo_Time": "None"}
# }
# """
    print("A_info_card", A_info_card)
    B_info_card = splitted_data[4].strip().replace(': None',': "None"')
    print("B_info_card", B_info_card)
    Discussions = splitted_data[6].strip().replace(': None',': "None"')
    A_json = json.loads(A_info_card)
    B_json = json.loads(B_info_card)
    discussion_json = json.loads(Discussions)
    
    #压缩key
    key_map = {
        "basic_information": "bi",
        "background": "bg",
        "others": "o",
        "name": "N",
        "gender": "Gd",
        "date_of_birth": "DB",
        "place_of_birth": "PB",
        "race": "R",
        "nationality": "Nat",
        "educational_background": "EB",
        "occupation": "Op",
        "position": "Pos",
        "achievement": "A",
        "personality": "Per",
        "hobbies": "H",
        "good_at": "G",
        "not_good_at": "Ng",
        "topics_of_interest": "Toi",
        "topics_of_disinterest": "Tod",
        "people_they_admire": "PTA",
        "people_they_dislike": "PTD",
        "todo": "Td",
        "todo_time": "TT",
        "topic": "T",
        "user_a's_opinion": "Ao",
        "user_b's_opinion": "Bo"
    }
    A_final = {}
    B_final = {}
    Discussions_final = {}
    for key0 in A_json.keys():
        A_final[key_map[key0.lower().replace(' ','_')]] = {}
        B_final[key_map[key0.lower().replace(' ','_')]] = {}
    for key0 in A_json.keys():
        for key1 in A_json[key0].keys():
            A_final[key_map[key0.lower().replace(' ','_')]][key_map[key1.lower().replace(' ','_')]] = A_json[key0][key1]
            B_final[key_map[key0.lower().replace(' ','_')]][key_map[key1.lower().replace(' ','_')]] = B_json[key0][key1]
    for key in discussion_json.keys():
        Discussions_final[key_map[key.lower().replace(' ','_')]] = discussion_json[key]
    return A_final, B_final, Discussions_final
    

    print("="*60)
    print("A信息卡：",A_final)
    print("="*60)
    print("="*60)
    print("B信息卡：",B_final)
    print("="*60)
    print("="*60)
    print("Discussion dic: ",Discussions_final)
    print("="*60)

# Replace `your-api-key` with your actual API key
def openai_chatcompletion(prompt, api_id=0):
    api_key = [
    # "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",
    # "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",
    # "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",
    # "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",
    "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",
    # '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'
    ]
    openai.api_key = api_key[api_id]
    cn = 0
    enc = tiktoken.get_encoding("cl100k_base")
    print(f"Tokens Of Prompt in {api_id}: {len(enc.encode(prompt))}!")
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

def concat_prompt(topic):
    themes_str = ""
    for i,theme in enumerate(themes):
        themes_str = themes_str+f"Theme{str(i+1)}: {theme}\n"

    content = f"""
    Classify the following topics into the most relevant theme.
    Requirement:
        1. If the topic is about personal information or personal experience description or interpersonal relationship, fill in "Class" with "Personal Information", else fill in "Class" with "Discussion".
        2. If the topic is in "Discussion" class,give the broad theme or category the topic falls into, such as politics, business, technology, health, sports, entertainment, etc..
        3. If a certain key's information is not mentioned, fill in it with "None".

    Topic:
    {topic}

    Output format:
        Class: xxx
        Theme/Category: xxx
    """
    return content

def PostProcess(response):
    if 'Class' in response and 'Theme/Category' in response:
        splitted_data = re.split(f"(Class|Theme/Category):", response, flags=re.I)
        class_of = splitted_data[2].strip() 
        theme = splitted_data[4].strip()
        # common_theme = splitted_data[6].strip()
        # new_theme = splitted_data[8].strip()
        print('SPLITTED_DATA', splitted_data)
        if "Personal Information" not in class_of:
            # if "None" not in common_theme:
            #     # print("COMMON THEME",common_theme)
            #     theme_num = int(common_theme.split('(Original theme: Theme')[-1].replace(')',''))
            #     # print('theme_num', theme_num)
            #     themes[theme_num-1] = common_theme.split('(Original theme: Theme')[0]
            # if "None" not in new_theme:
            #     themes.append(new_theme)
            # print("THEMES", themes)
            return theme
    return 

def load_sum_prompt(conv):
    prompt = """
    The following is a conversation between User_A and User_B. Please help to extract information and summarize opinion on both sides from the following conversation.
    Requirement:
    1. Divide the information into two parts, a description of people and opinions on a topic.
    2. Summarize the conversation into one topic and the opinion of both sides. Give both sides's  way of talking and did he achieve a achievement such as convincing someone, getting a message when discuss the topic using a concise sentence or some words.
    3. For the description of people, construct an information card of both sides.
    4. "Todo" is what people is going to do, "Todo_Time" is corresponding time.
    5. If a certain key's information is not mentioned, fill in it with "None".

    The structure of the information card is as follows:
    {"basic_information": {"Name": xxx, "Gender": xxx, "Date_of_Birth": xxx, "Place_of_Birth": xxx, "Race": xxx, "Nationality": xxx}, "background": {"Educational_Background": xxx, "Occupation": xxx, "Position": xxx, "Achievement": xxx}, "others": {"Personality": xxx, "Hobbies": xxx, "Good_at": xxx, "Not_good_at": xxx, "Topics_of_interest": xxx, "Topics_of_disinterest": xxx, "People_They_Admire": xxx, "People_They_Dislike": xxx, "Todo": xxx, "Todo_Time": xxx}}}

    The output format is as follows:
    User_A's information card: xxx

    User_B's information card: xxx

    Discussions:
    {"Topic": xxx,
    "Summary": xxx,
    "User_A's_opinion": xxx,
    "User_A's_way_of_talking": xxx,
    "User_A's_achievement":xxx,
    "User_B's_opinion": xxx,
    "User_B's_way_of_talking": xxx,
    "User_B's_achievement":xxx}

    The conversation is as follows:\n
    """
    prompt = prompt+conv
    return prompt

def load_eval_prompt(Answer1, Answer2, conv):
    prompt = f"""
    We hope you can provide feedback on the performance of the two AI assistants to answer the user question displayed below. Assuming Answer1 is 80 points, based on the relevance of the answer to the entire conversation and the amount of information in the Answer2, give the Answer2 a score of 0-100 points. Please provide a comprehensive reason for your evaluation, avoid any potential biases, and ensure that the order of responses does not affect your judgment. The format is as follows:  \nCore: xxx  \nReason: xxx
    
    Question: The following is a conversation between User_A and User_B. Please help to extract information and summarize opinion on both sides from the following conversation.
    
    The conversation is:
        {conv}
        
    Answer1: 
        {Answer1}

    Answer2:
        {Answer2}

    """
    return prompt



def load_group_sum_prompt(conv):
    prompt = """The following conversation is a group conversation. Please summarize the opinions of all parties.
    Requirement:
    1. Provide a summary and a topic of the discussion. Give everyone's opinion on the topic.
    2. Provide the outcome of this discussion.
    3. Speculate the roles played by each member in the group chat during the discussion.

    The output format is as follows:
    Topic: xxx
    Summary: xxx
    Opinion:
        A: xxx
        B: xxx
        ...
    Outcome: xxx
    Role:
        A: xxx
        B: xxx
        ...


    The conversation is as follows:
    [Start of conversation]"""
    prompt = prompt+conv.strip()+"\n[End of conversation]"
    return prompt

if __name__ == "__main__":
    conv = """"A : so we've got new project requirements .",
        "A : So basically we've got three things ,",
        "A : and we've got forty minutes in which to uh <disfmarker> for this meeting to uh to discuss the various options .",
        "A : W I I got um <gap> or or three things basically , um relating to the remote being only for T_V_ .",
        "A : We discussed that last time",
        "A : Um we've got uh teletext outdated .",
        "A : Right and the corporate image was the uh final thing .",
        "A : So I I got that in email form .",
        "B : Okay I'm gonna be looking at the working design .",
        "B : Um I've just got three sections ,",
        "B : first is the research I made on the on the remote control itself um .",
        "B : And then that involves the components required in it and the systems uh design of the actual the actual remote .",
        "B : Um so having researched the existing models within the market , um I found my research off the internet . Um I've established what the components required for the remote control to function , actually are .",
        "B : And then also the methods in which these components interact together for the remote to actually do what you want it to do",
        "B : Um the basic components are an energy source",
        "B : which I guess um in most existing models would be a battery supply .",
        "B : We then have the user interface , which is basically the like the the buttons on the actual remote .",
        "B : Um the various functions used for changing channel , uh channel up and down , volume , things like that .",
        "B : Um there's also a chip inside the remote which does all the computer type things .",
        "B : And then the sender , which um is usually , I've found , an infra-red device which sends a signal to the actual television .",
        "B : Um and the last part is receiver which is important in the system",
        "B : because that's obviously found in the television .",
        "B : Um I'm gonna have to actually draw on the board",
        "B : because uh it was a little tricky on PowerPoint to get this working ,",
        "B : a power supply",
        "B : we then have a particular button ,",
        "B : <gap> that's obviously there's lots and lots of different buttons .",
        "B : after you press that that sends the message to the chip ,",
        "B : and then sends the appropriate message to the sender . <vocalsound>",
        "B : that's the components of the remote",
        "B : So this is the uh user interface .",
        "B : And then on the separate thing we have on the on the television we have a a receiver .",
        "B : which then <gap> , and that's the that's the infra-red sender .",
        "B : <gap> going on to personal preferences , I've said that battery seems the best option for the actual remote ,",
        "B : just because of the size .",
        "B : and infra-red um has been used quite successfully .",
        "B : Um and then the sender ,",
        "C : Um . Okay so I'm gonna talk a bit about the technical functions design .",
        "C : <vocalsound> Um so the m basic method of this is to send a signal from the remote to the television set , so that a desired function is performed .",
        "C : Um here are two example remotes .",
        "C : Um by the look of it they both have um kind of play and fast forward , rewind functions ,",
        "C : so I think they incorporate a kind of video function which we won't have to worry about .",
        "C : Uh but as you can see , the left remote is quite um quite busy looking , quite complicated .",
        "C : Um whereas the right remote is much simpler ,",
        "C : it looks much more user friendly .",
        "C : Um so my personal preference would be the right remote .",
        "C : it's got a very limited number of buttons .",
        "C : So , <vocalsound> it's got nice big buttons ,",
        "C : Um I like the use of the kind of um symbols like the triangles and the squares and the arrows as well as the words on the um kind of play functions and all that .",
        "C : So it's very very user friendly ,",
        "A : <vocalsound> But uh I got uh I got an email that basically said to uh make sure that uh whatever device we come up with at the end of the day had to incorporate um the corporate colour and slogan .",
        "A : Um we have to remember that we have our own um logo and colour scheme .",
        "B : Would you be able to get rid of the the extra buttons here , the the sort of circular section ,",
        "B : So we could dispense with that little bit as well",
        "B : because that seems to be for a video as well .",
        "B : and just get it down to just the numbers and the volume .",
        "C : And I don't really think that you need nine numbers .",
        "C : I mean how often do you use seven , eight and nine ?",
        "C : I think just one to six and then channel up and down should be enough .","""
    print('-'*60)
    print("PROMPT:\n", load_group_sum_prompt(conv))
    print('-'*60)
    print(openai_chatcompletion(load_group_sum_prompt(conv)))
