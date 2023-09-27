"""LTM database"""

import pathlib
import sqlite3
from typing import Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.embeddings.huggingface import HuggingFaceInstructEmbeddings
from sklearn.neighbors import NearestNeighbors
import zarr

from extensions.long_term_memory.constants import (
    CHUNK_SIZE,
    DATABASE_NAME,
    EMBEDDINGS_NAME,
    EMBEDDING_VECTOR_LENGTH,
    SENTENCE_TRANSFORMER_MODEL,
)
from extensions.long_term_memory.core.queries import (
    CREATE_TABLE_QUERY,
    DROP_TABLE_QUERY,
    FETCH_DATA_QUERY,
    INSERT_DATA_QUERY,
)

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import VectorStoreRetrieverMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.docstore import InMemoryDocstore
from langchain.vectorstores import FAISS

import faiss
import os
import time

os.environ["OPENAI_API_KEY"] = "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"

class LtmDatabase:
    """API over an LTM database."""

    def __init__(
        self,
        directory: pathlib.Path,
        num_memories_to_fetch: int=1,
        force_use_legacy_db: bool=False,
    ):
        """Loads all resources."""
        self.directory = directory

        self.database_path = None
        self.embeddings_path = None

        self.character_name = None
        self.message_embeddings = None
        self.disk_embeddings = None
        self.sql_conn = None

        self.vectorstore=None
        self.memory=None

        # Load db
        (legacy_database_path, legacy_embeddings_path) = self._build_database_paths()
        legacy_db_exists = legacy_database_path.exists() and legacy_embeddings_path.exists()
        use_legacy_db = force_use_legacy_db or legacy_db_exists
        if use_legacy_db:
            print("="*20)
            print("WARNING: LEGACY DATABASE DETECTED, CHARACTER NAMESPACING IS DISABLED")
            print("         See README for character namespace migration instructions if you want different memories for different characters")
            print("="*20)
            self.database_path = legacy_database_path
            self.embeddings_path = legacy_embeddings_path
            self._load_db()

        self.use_legacy_db = use_legacy_db
        

    def _build_database_paths(self, character_name: Optional[str]=None):
        database_path = self.directory / DATABASE_NAME \
                if character_name is None \
                else self.directory / character_name / DATABASE_NAME
        embeddings_path = self.directory / EMBEDDINGS_NAME \
                if character_name is None \
                else self.directory / character_name / EMBEDDINGS_NAME

        return (database_path, embeddings_path)

    def _load_db(
        self,
        database_namespace: str="LEGACY_UNIFIED_DATABASE",
    ):
        if not self.database_path.exists() and not self.embeddings_path.exists():
            print(f"No existing memories found for {database_namespace}, "
                  "will create a new database.")
            self._destroy_and_recreate_database(do_sql_drop=False)
        elif self.database_path.exists() and not self.embeddings_path.exists():
            raise RuntimeError(
                f"ERROR: Inconsistent state detected for {database_namespace}: "
                f"{self.database_path} exists but {self.embeddings_path} does not. "
                "Her memories are likely safe, but you'll have to regen the "
                "embedding vectors yourself manually."
            )
        elif not self.database_path.exists() and self.embeddings_path.exists():
            raise RuntimeError(
                f"ERROR: Inconsistent state detected for {database_namespace}: "
                f"{self.embeddings_path} exists but {self.database_path} does not. "
                f"Please look for {DATABASE_NAME} in another directory, "
                "if you can't find it, her memories may be lost."
            )

        ### Prepare the memory database for retrieve ###
        # Load the embeddings to a local numpy array
        # lw change 
        # self.message_embeddings = zarr.open(self.embeddings_path, mode="r")[:]
        # # Prepare a "connection" to the embeddings, but to store new LTMs on disk
        # self.disk_embeddings = zarr.open(self.embeddings_path, mode="a")
        # # Prepare a "connection" to the master database
        # self.sql_conn = sqlite3.connect(self.database_path, check_same_thread=False)
        # embeddings = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-xl')#zmy
        # self.message_embeddings = FAISS.load_local(self.database_path, embeddings, "database")
        self.message_embeddings = FAISS.load_local(self.database_path, OpenAIEmbeddings(), "database")

        # self.disk_embeddings = FAISS.load_local(self.embeddings_path, embeddings, "embeddings")
        self.disk_embeddings = FAISS.load_local(self.embeddings_path, OpenAIEmbeddings(), "embeddings")

        retriever = self.disk_embeddings.as_retriever(search_kwargs=dict(k=3)) #zmy k=5
        self.memory = VectorStoreRetrieverMemory(retriever=retriever)


    def _destroy_and_recreate_database(self, do_sql_drop=False) -> None:
        """Destroys and re-creates a new LTM database.

        WARNING: THIS WILL DESTROY ANY EXISTING LONG TERM MEMORY DATABASE.
                 DO NOT CALL THIS METHOD YOURSELF UNLESS YOU KNOW EXACTLY
                 WHAT YOU'RE DOING!
        """
        # Create directories if they don't exist
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

        # Create new sqlite table to store the textual memories
        #lw change
        # sql_conn = sqlite3.connect(self.database_path)
        # with sql_conn:
        #     if do_sql_drop:
        #         sql_conn.execute(DROP_TABLE_QUERY)
        #     sql_conn.execute(CREATE_TABLE_QUERY)

        # # Create new embeddings db to store the fuzzy keys for the
        # # corresponding memory text.
        # # WARNING: will destroy any existing embeddings db
        # zarr.open(
        #     self.embeddings_path,
        #     mode="w",
        #     shape=(0, EMBEDDING_VECTOR_LENGTH),
        #     chunks=(CHUNK_SIZE, EMBEDDING_VECTOR_LENGTH),
        #     dtype="float32",
        # )

        embedding_size = 1536 # Dimensions of the OpenAIEmbeddings
        index = faiss.IndexFlatL2(embedding_size)
        embedding_fn = OpenAIEmbeddings().embed_query
        # embedding_fn = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-xl')#zmy
        self.vectorstore = FAISS(embedding_fn, index, InMemoryDocstore({}), {})

        self.vectorstore.save_local(self.database_path, "database")
        self.vectorstore.save_local(self.embeddings_path, "embeddings")

    def load_character_db_if_new(self, character_name: str) -> None:
        """Loads the database associated with the specified character."""
        if self.use_legacy_db:
            # Using legacy database, do nothing
            return
        if self.character_name == character_name:
            # No change in character, do nothing.
            return

        print(f"loading character {character_name}")

        # Load db of new character.
        (self.database_path, self.embeddings_path) = self._build_database_paths(character_name)
        self._load_db(character_name)
        self.character_name = character_name
    
    def load_path_db_if_new(self, mem_root: pathlib.Path) -> None:
        """Loads the database associated with the specified character."""
        if self.use_legacy_db:
            # Using legacy database, do nothing
            return
        if self.directory == mem_root:
            # No change in character, do nothing.
            return
        print('old directory:',self.directory,type(self.directory))
        print(f"loading memory path {mem_root},type:{type(mem_root)}")

        # Load db of new character.
        self.directory = mem_root
        (self.database_path, self.embeddings_path) = self._build_database_paths(self.character_name)
        self._load_db(self.character_name)
        # self.character_name = character_name

    # #Lw change  
    # def add(self, name: str, new_message: str, name2: str, new_message2: str) -> None:
    #     """Adds a single new sentence to the LTM database."""
    #     # Create the message embedding
    #     c=time.localtime()
    #     d=time.strftime("%Y-%m-%d %H:%M:%S",c)
    #     print("lw debug time",d)
    #     self.memory.save_context({name2: new_message2}, {name:new_message+" timestamp: {}".format(d)}) #, {"timestamp":d})
    #     self.message_embeddings.save_local(self.database_path, "database")
    #     self.disk_embeddings.save_local(self.embeddings_path, "embeddings")

    #Zmy change
    def add(self, name: str, new_message: str) -> None: #name:日期-轮数 new_message:摘要（有固定格式）
        """Adds a single new sentence to the LTM database."""
        # Create the message embedding
        new_message = new_message[0]['summary_text']
        print("zmy debug name","{}".format(name))
        self.memory.save_context({"{}".format(name): new_message}, {}) #, {"timestamp":d})
        self.message_embeddings.save_local(self.database_path, "database")
        self.disk_embeddings.save_local(self.embeddings_path, "embeddings")
        print('zmy-mem_path',self.database_path)


    # def query(self, query_text: str) -> List[Tuple[Dict[str, str], float]]:
    #     """Queries for the most similar sentence from the LTM database."""
    #     # If too few LTM features are loaded, return nothing.
    #     if self.message_embeddings.shape[0] == 0:
    #         return []

    #     # Create the query embedding
    #     query_text_embedding = self.sentence_embedder.encode(query_text)
    #     query_text_embedding = np.expand_dims(query_text_embedding, axis=0)

    #     # Find the most relevant memory's index with FAISS
        
        
    #     embedding_searcher = NearestNeighbors(
    #         n_neighbors=min(self.num_memories_to_fetch, self.message_embeddings.shape[0]),
    #         algorithm="brute",
    #         metric="cosine",
    #         n_jobs=-1,
    #     )
    #     embedding_searcher.fit(self.message_embeddings)
    #     (match_scores, embedding_indices) = embedding_searcher.kneighbors(
    #         query_text_embedding
    #     )

    #     all_query_responses = []
    #     for (match_score, embedding_index) in zip(match_scores[0], embedding_indices[0]):
    #         with self.sql_conn as cursor:
    #             response = cursor.execute(FETCH_DATA_QUERY, (int(embedding_index),))
    #             (name, message, timestamp) = response.fetchone()

    #         query_response = {
    #             "name": name,
    #             "message": message,
    #             "timestamp": timestamp,
    #         }
    #         all_query_responses.append((query_response, match_score))

    #     return all_query_responses

    # def query(self, query_text: str, name: str) -> List[Tuple[Dict[str, str], float]]:
    #     """Queries for the most similar sentence from the LTM database."""
        
    #     # Create the query embedding
    #     query_responses=self.memory.load_memory_variables({"prompt": query_text})["history"]
    #     all_query_responses = []
    #     query_responses = query_responses.split(name+": ")
    #     print("lw debug query_responses",query_responses)
    #     for query_response in query_responses:
    #         if query_response=="":continue
    #         print("lw debug query responce",query_response)
    #         query_response = name+": "+query_response
    #         query_response = query_response.split(" timestamp: ")
    #         name_save="all"
    #         message=query_response[0]
    #         timestamp=query_response[1].strip()
    #         query_response = {
    #             "name": name,
    #             "message": message,
    #             "timestamp": timestamp,
    #         }
    #         match_score=1
    #         all_query_responses.append((query_response, match_score))

    #ZMY Versoin
    def query(self, query_text: str) -> List[Tuple[Dict[str, str], float]]:
        """Queries for the most similar sentence from the LTM database."""
        all_query_responses = []
        # Create the query embedding
        query_responses=self.memory.load_memory_variables({"prompt": query_text})["history"]
    
        match_score = 1
        query_responses = query_responses.split("sum_round")
        

        for i in range(1,len(query_responses)):
            if query_responses != '':
                name = query_responses[i].split(":")[0]
                timestamp = query_responses[i-1].split('\n')[-1][:-1]
                query_response = {
                    "sum_round":int(name),
                    "message":query_responses[i][len(name)+1:-(len(timestamp))],
                    "timestamp":timestamp
                }
                all_query_responses.append((query_response, match_score))
        print('zmy-mem_path',self.database_path)
        # print('zmy-query-response',all_query_responses)
        # return query_response
        # all_query_responses = []
        # query_responses = query_responses.split(name+": ")
        # print("lw debug query_responses",query_responses)
        # for query_response in query_responses:
        #     if query_response=="":continue
        #     print("lw debug query responce",query_response)
        #     query_response = name+": "+query_response
        #     query_response = query_response.split(" timestamp: ")
        #     name_save="all"
        #     message=query_response[0]
        #     timestamp=query_response[1].strip()
        #     query_response = {
        #         "name": name,
        #         "message": message,
        #         "timestamp": timestamp,
        #     }
        #     match_score=1
        #     all_query_responses.append((query_response, match_score))


        # query_text_embedding = self.sentence_embedder.encode(query_text)
        # query_text_embedding = np.expand_dims(query_text_embedding, axis=0)

        # # Find the most relevant memory's index with FAISS
        
        
        # embedding_searcher = NearestNeighbors(
        #     n_neighbors=min(self.num_memories_to_fetch, self.message_embeddings.shape[0]),
        #     algorithm="brute",
        #     metric="cosine",
        #     n_jobs=-1,
        # )
        # embedding_searcher.fit(self.message_embeddings)
        # (match_scores, embedding_indices) = embedding_searcher.kneighbors(
        #     query_text_embedding
        # )

        # all_query_responses = []
        # for (match_score, embedding_index) in zip(match_scores[0], embedding_indices[0]):
        #     with self.sql_conn as cursor:
        #         response = cursor.execute(FETCH_DATA_QUERY, (int(embedding_index),))
        #         (name, message, timestamp) = response.fetchone()

        #     query_response = {
        #         "name": name,
        #         "message": message,
        #         "timestamp": timestamp,
        #     }
        #     all_query_responses.append((query_response, match_score))

        return all_query_responses

    def reload_embeddings_from_disk(self) -> None:
        """Reloads all embeddings from disk into memory."""
        if self.message_embeddings is None:
            return

        print("--------------------------------")
        print("Loading all embeddings from disk")
        print("--------------------------------")
        num_prior_embeddings = self.message_embeddings.shape[0]
        self.message_embeddings = zarr.open(self.embeddings_path, mode="r")[:]
        num_curr_embeddings = self.message_embeddings.shape[0]
        print("DONE!")
        print(f"Before: {num_prior_embeddings} embeddings in memory")
        print(f"After: {num_curr_embeddings} embeddings in memory")
        print("--------------------------------")

    def destroy_all_memories(self) -> None:
        """Deletes all embeddings from memory AND disk."""
        if self.message_embeddings is None or self.disk_embeddings is None:
            return

        print("--------------------------------------------------")
        print("Destroying all memories, I hope you backed them up")
        print("--------------------------------------------------")
        self.message_embeddings = None
        self.disk_embeddings = None

        self._destroy_and_recreate_database(do_sql_drop=True)

        self.disk_embeddings = zarr.open(self.embeddings_path, mode="a")
        self.message_embeddings = zarr.open(self.embeddings_path, mode="r")[:]
        print("DONE!")
        print("--------------------------------------------------")
