import torch
import argparse

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig, Pipeline
from langchain_community.vectorstores import FAISS

from config import Config

from knowledge_base import load_knowledge_vdbase

def create_reader_pipeline(config: Config, return_tokenizer=False):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    model = AutoModelForCausalLM.from_pretrained(config.READER_MODEL_NAME, quantization_config=bnb_config)
    tokenizer = AutoTokenizer.from_pretrained(config.READER_MODEL_NAME)
    
    READER_LLM = pipeline(
        model=model,
        tokenizer=tokenizer,
        task='text-generation',
        do_sample=True,
        temperature=0.2,
        repetition_penalty=1.1,
        return_full_text=False,
        max_new_tokens=config.MAX_NEW_TOKENS,
    )
    
    if return_tokenizer:
        return READER_LLM, tokenizer
    
    return READER_LLM


def get_prompt_template(tokenizer):
    prompt_in_chat_format = [
    {
        "role": "system",
        "content": """Using the information contained in the context,
        give a comprehensive answer to the question.
        Respond only to the question asked, response should be concise and relevant to the question.
        Provide the number of the source document when relevant.
        If the answer cannot be deduced from the context, do not give an answer.""",
            },
            {
                "role": "user",
                "content": """Context:
        {context}
        ---
        Now here is the question you need to answer.

        Question: {question}""",
            },
        ]
    
    rag_prompt_template = tokenizer.apply_chat_template(
        prompt_in_chat_format, tokenize=False, add_generation_prompt=True
    )
    
    return rag_prompt_template


def answer_with_rag(
    question: str,
    llm: Pipeline,
    knowledge_base : FAISS,
    prompt_template,
    num_docs_fetched=5
):
    print("=> Retrieving documents...")
    fetched_docs = knowledge_base.similarity_search(question, k=num_docs_fetched)
    fetched_docs = [doc.page_content for doc in fetched_docs]
    
    print("=> Extracting text...")
    context = "\nExtracted documents:\n"
    context += "".join([f"Document {str(i)}:::\n" + doc for i, doc in enumerate(fetched_docs)])
    
    prompt = prompt_template.format(question=question, context=context)
    
    print("=> Generating answer...")
    answer = llm(prompt)[0]['generated_text']
    
    print("=> Done!")
    
    return answer, fetched_docs


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(prog='Getting answer with RAG.')
    
    parser.add_argument("question", type=str)
    parser.add_argument("dbase_path", type=str)
    
    args = parser.parse_args()
    
    config = Config()
    
    reader, tokenizer = create_reader_pipeline(config, return_tokenizer=True)
    
    knowledge_dbase = load_knowledge_vdbase(args.dbase_path, config=config)
    
    prompt_template = get_prompt_template(tokenizer)
    
    answer, _ = answer_with_rag(args.question, reader, knowledge_dbase, prompt_template)
    
    print(answer)
    
    