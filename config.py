from dataclasses import dataclass


@dataclass
class Config:
    # Dataset name
    DATASET_NAME = "m-ric/huggingface_doc"
    
    # Separators for document chunking
    MARKDOWN_SEPARATORS = [
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    "",
    ]
    # Size of every document's chunk
    CHUNK_SIZE = 512
    
    # Model name for documents' embeddings
    EMBEDDING_MODEL_NAME = 'thenlper/gte-small'
    
    # Reader-LLM name from Hugging face
    READER_MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"
    
    # Maximum amount of tokens in model answer
    MAX_NEW_TOKENS = 700
    
    
    
    
     