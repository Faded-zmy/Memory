import json
import os
from langchain import OpenAI 
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationKGMemory
from MyChain import MyChain
os.environ["OPENAI_API_KEY"] = "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"
llm = OpenAI(temperature=0)
template = """The following is a friendly conversation between some members and an AI. The AI is talkative and provides lots of specific details from its context. 
If the AI does not know the answer to a question, it truthfully says it does not know. The AI ONLY uses information contained in the "Relevant Information" section and does not hallucinate.

Relevant Information:

{history}

Conversation:
{Member}: {Member_message}
AI:"""

prompt = PromptTemplate(
    input_variables=["history", "Member", "Member2"], template=template
)
# prompt = PromptTemplate.from_template(
#     template=template
# )
print('prompt',prompt)
conversation_with_kg = MyChain(
    llm=llm, 
    verbose=True, 
    prompt=prompt,
    memory=ConversationKGMemory(llm=llm),
    input_key=["Member1", "Member2"]
)
conv = json.load(open('../generate_data/conversation_data/conversation_0.json','r'))[0]['conversation'].split('\n\n')[0].split('\n')
# print('conv',conv)
# mem_conv={
#     "Member1": "Hey guys, my birthday is coming up next week!",
#     "Member2": "That's great, how are you planning to celebrate?"
#     }
# conversation_with_kg.load_memory_variables(mem_conv)
for i,c in enumerate(conv):
    if c.split(':')[0] == "Question":
        # conv_above='\n'.join(conv[:i])
        input_q=':'.join(c.split(':')[1:])
        conversation_with_kg.predict(input=input_q)
        print(c)
        print(conversation_with_kg.predict(input="What type of cake is the birthday Member1 planning to have? "))
        # break
    elif c.split(':')[0] == "Answer":
        print(c)
        pass
    else:
        print(c)
        input_q=c.split(':')[0]+' says'+':'.join(c.split(':')[1:])
        print(conversation_with_kg.input_keys)
        conversation_with_kg.predict(Member1=input_q)


# print('conv_above',conv_above)
print('input',input_q)
# print(conversation_with_kg.predict(history=conv_above,input="What type of cake is the birthday Member1 planning to have? "))
# print(conversation_with_kg.predict(input="hello"))