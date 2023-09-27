from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
# from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationSummaryMemory
from langchain.chains.conversation.memory import ConversationKGMemory

import json
import os

os.environ["OPENAI_API_KEY"] = "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"

#load对话及问题
conv = json.load(open('../generate_data/conversation_data/conversation_0.json','r'))[0]['conversation'].split('\n\n')[0].split('\n')
print('conv',conv)
#embedding类型
embeddings = OpenAIEmbeddings()

#memory类型
llm = OpenAI(temperature=0)
memory = ConversationSummaryMemory(llm=llm,memory_key="chat_history", return_messages=True)

#逐句储存，遇到问题，根据上面的对话回答
print('conv[0]',conv[0].replace(':',' says:"')+'"')
vec_db = FAISS.from_texts(texts=[conv[0].replace(':','says:"')+'"'], embedding=embeddings)#初始化vector database
for sentence in conv[1:]:
    if sentence.split(':')[0] not in ['Question','Answer']:
        sentence = sentence.replace(':',' says:"')+'"'
        vec = FAISS.from_texts(texts=[sentence], embedding=embeddings)
        vec_db.merge_from(vec)
    elif sentence.split(':')[0] == 'Question':
        print(sentence)
        query=sentence.replace('Question:','')
        qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), vec_db.as_retriever(), memory=memory)
        # print('similarity_search',vec_db._similarity_search_with_relevance_scores(query=query,k=3))
        result = qa({"question": query})['answer']
        print('result',result)
    elif sentence.split(':')[0] == 'Answer':
        print(sentence)

#保存database
vec_db.save_local("faiss_index")
#load database
db = FAISS.load_local("faiss_index", embeddings)







