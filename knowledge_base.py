import argparse

from tqdm import tqdm

from transformers import AutoTokenizer
from datasets import load_dataset

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

from config import Config


def get_raw_knowledge_base(dataset):
    return [Document(page_content=doc["text"], metadata={"source": doc["source"]}) for doc in tqdm(dataset)]


def split_documents(config:Config, knowledge_base):
    chunker = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=AutoTokenizer.from_pretrained(config.EMBEDDING_MODEL_NAME),
        separators=config.MARKDOWN_SEPARATORS,
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=int(config.CHUNK_SIZE / 10),
        strip_whitespace=True,
        add_start_index=True
    )
    
    docs_processed = []
    for doc in knowledge_base:
        docs_processed += chunker.split_documents([doc])
    
    unique_texts = {}
    docs_processed_unique = []
    for doc in docs_processed:
        if doc.page_content not in unique_texts:
            unique_texts[doc.page_content] = True
            docs_processed_unique.append(doc)
    
    return docs_processed_unique


def get_embedding_model(config):
    embedding_model = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME,
        multi_process=True,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True}
    )
    
    return embedding_model


def create_knowledge_vdbase(
    config : Config, 
    save_local=True, 
    folder_path="", 
    docs_processed=None, 
    dataset=None
    ):
    embedding_model = get_embedding_model(config)
    
    assert docs_processed is not None or dataset is not None, "Either docs or dataset should be given."
    
    if docs_processed is None:
        print("=> Getting raw knowledge base from dataset...")
        raw_knowledge_base = get_raw_knowledge_base(dataset)
        
        print("=> Chunking knowledge base into documents...")
        docs_processed = split_documents(config, raw_knowledge_base)
    
    print("=> Creating knowledge vector base...")
    knowledge_vector_base = FAISS.from_documents(
        docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE
    )
    
    if save_local:
        print("=> Saving locally...")
        knowledge_vector_base.save_local(folder_path=folder_path)
        
        print(f"=> Knowledge vector base was saved locally in folder {folder_path}.")
        
    return knowledge_vector_base


def load_knowledge_vdbase(path: str, config: Config):
    
    embedding_model = get_embedding_model(config)
    
    knowledge_vdbase = FAISS.load_local(folder_path=path, embeddings=embedding_model, allow_dangerous_deserialization=True)
    
    return knowledge_vdbase
    

if __name__ == '__main__':
    
    config = Config()
    
    parser = argparse.ArgumentParser(prog='Knowledge vector base creation.')
    
    parser.add_argument("--folder_path", const="", nargs='?')
    
    args = parser.parse_args()
    
    dataset = load_dataset(config.DATASET_NAME, split='train')
    
    create_knowledge_vdbase(config, dataset=dataset, save_local=True, folder_path=args.folder_path)