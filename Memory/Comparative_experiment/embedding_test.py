from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
# from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationSummaryMemory
from langchain.chains.conversation.memory import ConversationKGMemory

import json
import os
import utils

# def get_response_from_gpt(contexts,query):
#     prompt = f"Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.\n{contexts}\nQuestion: {query}\nHelpful Answer:"
#     print('prompt',prompt)
#     result = utils.openai_chatcompletion2(prompt)
#     return result


os.environ["OPENAI_API_KEY"] = "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"

#load对话及问题
# conv = json.load(open('../generate_data/conversation_data/conversation_0.json','r'))[2]['conversation'].split('\n\n')[0].split('\n')
# print('conv',conv)


#memory类型
# llm = OpenAI(temperature=0)
# memory = ConversationSummaryMemory(llm=llm,memory_key="chat_history", return_messages=True)

#逐句储存，遇到问题，根据上面的对话回答
conv = [
    " Person A asks about traveling to Japan, and Person B shares their past experience of visiting the country.",
    "Person A mentions a recent soccer match and the score, and Person B expresses their interest in the overall performance of the teams.",
    "Person A shares their success with a new chocolate cake recipe, and Person B asks about the ingredients used.",
    "Person A mentions reading a thought-provoking novel, and Person B expresses their interest in the central theme of the book.",
    "Person A shares their positive experience with a daily yoga practice, and Person B asks about the specific yoga poses.",
    "Person A mentions watching a recent action movie with impressive special effects, and Person B shows interest in the overall plot.",
    "Person A talks about their successful tomato gardening, and Person B asks about the care and maintenance of the plants.",
    "Person A mentions a new smartphone with an advanced camera, and Person B expresses curiosity about other features.",
    " Person A shares their guitar learning journey, and Person B asks about the songs they have been practicing.",
    "Person A talks about their experience at a jazz concert and praises the talented musicians, while Person B inquires about the instruments showcased."
    
]
query = " do exercise"
print('conv[0]',conv[0])
#embedding类型
embedding_ls = {
    'openai':OpenAIEmbeddings(), 
    'hkunlp/instructor-xl':HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-xl'), 
    'sentence-transformers/all-MiniLM-L6-v2':HuggingFaceInstructEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2'),
    'sentence-transformers/all-mpnet-base-v2':HuggingFaceInstructEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'),
    'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2':HuggingFaceInstructEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

    }
for embedding_name in embedding_ls.keys():  
    embeddings = embedding_ls[embedding_name]
    vec_db = FAISS.from_texts(texts=[conv[0]], embedding=embeddings)#初始化vector database
  
    for sentence in conv[1:]:
        vec = FAISS.from_texts(texts=[sentence], embedding=embeddings)
        vec_db.merge_from(vec)
        
    contexts_lc = vec_db._similarity_search_with_relevance_scores(query=query,k=3)
    print("="*60)
    print('Model:{}'.format(embedding_name))
    print('-'*60)
    print(contexts_lc)


# #保存database
# vec_db.save_local("faiss_index")
# #load database
# db = FAISS.load_local("faiss_index", embeddings)







