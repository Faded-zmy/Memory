from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationSummaryMemory
from langchain.chains.conversation.memory import ConversationKGMemory
from langchain import OpenAI,VectorDBQA

import json
import os

os.environ["OPENAI_API_KEY"] = "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"
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
conv = json.load(open('../generate_data/conversation_data/conversation_0.json','r'))[0]['conversation'].split('\n\n')[0]
conv_above = conv.split('Question:')[0].replace(':',' says:"').replace('\n','" ')
print('conv',conv_above)
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_texts(texts=[conv_above], embedding_function=embeddings, persist_directory="./vector_store")
vectorstore.persist()
# conv_after = "Member4: I've actually been to Japan too, but I went to Kyoto instead.\nMember2: How was it? I've been wanting to visit there as well.\nMember4: It was beautiful! We visited a lot of shrines and temples and even saw some geishas.\nMember1: That sounds amazing. Did you try any local food?\nMember4: Yes, we had some amazing sushi and also tried some matcha sweets.\nMember3: I've heard the bamboo forest in Kyoto is really pretty. Did you get a chance to visit?\nMember4: Yes, we went early in the morning and it was really peaceful. Highly recommend it!\n"
# vectorstore = Chroma.from_texts(texts=[conv_after], embedding_function=embeddings, persist_directory="./vector_store")
# vectorstore.persist()
print('vector_store',type(vectorstore))


vectorstore = Chroma(persist_directory="./vector_store", embedding_function=embeddings)
llm = OpenAI(temperature=0)
memory = ConversationKGMemory(llm=llm,memory_key="chat_history", return_messages=True)
# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), vectorstore.as_retriever(), memory=memory)
query = "What museum did Member2 visit in Japan?"
# 进行问答
result = qa({"question": query})
print(result)


