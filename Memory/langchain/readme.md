## 各个脚本差别

# 根据聊天记录（文档）回答问题
* chat_with_conv_history_faiss.py
原始用langchain中的组件，包括OpenAI的embedding \Memory \retrieval库（faiss in langchain）\RerievalChain

* chat_with_conv_history_faiss_withoutOpenAI.py
将RetrievalChain换成：在库里（vector_database）搜索相关的context，加上问题以及template作为prompt给chatgpt生成答案（效果不错）


