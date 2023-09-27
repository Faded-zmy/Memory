import utils
from langchain.embeddings.huggingface import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS

def get_response_from_gpt(contexts,query):
    prompt = f"Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.\n{contexts}\nQuestion: {query}\nHelpful Answer:"
    print('prompt',prompt)
    result = utils.openai_chatcompletion2(prompt)
    return result

#embedding类型
embeddings = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-xl')
#生成embedding放入vector database中
vec_db = FAISS.from_texts(texts=["存入memory的内容"], embedding=embeddings)
#新存入memory的内容生成embedding,并存入vector database中
vec = FAISS.from_texts(texts=[sentence], embedding=embeddings)
vec_db.merge_from(vec)
#在vector_db中搜索question相近的k个vector,对应的内容作为contexts
contexts_lc = vec_db._similarity_search_with_relevance_scores(query=query,k=7)
#将contexts以及问题塞进模型（chatgpt）得到答案
result = get_response_from_gpt(contexts,query)

#保存database
vec_db.save_local("faiss_index")
#load database
db = FAISS.load_local("faiss_index", embeddings)