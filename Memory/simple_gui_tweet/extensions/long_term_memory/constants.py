"""Shared constants."""
import modules.shared as shared
import os
# Embedding-related Constants
EMBEDDING_VECTOR_LENGTH = 768
CHUNK_SIZE = 1000

# File Paths
# DATABASE_NAME = "long_term_memory.faiss"
# EMBEDDINGS_NAME = "long_term_memory_embeddings.faiss"
DATABASE_NAME = "long_term_memory_sum.faiss"
EMBEDDINGS_NAME = "long_term_memory_embeddings_sum.faiss"
# TMP_CONV = "./extensions/long_term_memory/user_data/bot_memories/tmp_conv.json"#存放最后几轮（不足固定轮数）的对话
# ORIGINAL_CONV = "./extensions/long_term_memory/user_data/bot_memories/original_conv/"#存放原对话
if hasattr(shared, 'mem_root'):
    _MEM_ROOT = shared.mem_root
    if not os.path.exists(mem_root):
        os.system('mkdir {}'.format(mem_root))
else:
    _MEM_ROOT = "extensions/long_term_memory/user_data/bot_memories"
# ORIGINAL_CONV = _MEM_ROOT+'/original_conv/'
# Hugging Face Models
SENTENCE_TRANSFORMER_MODEL = "sentence-transformers/all-mpnet-base-v2"


#总结摘要的轮数
# _CONVERSATION_ROUND_TO_SUM = 5
#总结摘要的单词数\n overlapped的对话轮数
_CONVERSATION_WORDS_TO_SUM = 200
_OVERLAPPED_ROUNDS = 2

