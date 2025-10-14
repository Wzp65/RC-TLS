import os
from tqdm import tqdm
import json

from transformers import AutoTokenizer, AutoModel

from chromadb import PersistentClient
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from utils import get_sentence_embedding, get_contextual_embedding
import torch


def create_db_word(train_data, tokenizer, model, device):
    # 初始化 ChromaDB 客户端
    client = PersistentClient()
    try:
        collection = client.get_collection(name='maven_contextual_words_bert_large')
    except:
        # 创建或获取集合（Collection）
        collection = client.create_collection(
            name="maven_contextual_words_bert_large",
            embedding_function=None,  # 禁用默认嵌入
            metadata={"hnsw:space": "cosine"}
        )
        
        embeddings = []
        metadatas = []
        documents = []
        ids = []  # ChromaDB 1.0+ 需要显式指定 ID（可选）

        idx = 0
        for data in tqdm(train_data, desc="create word_db"):
            sentence = data["sentence"]
            for item in data["trigger_info"]["triggers"]:
                trigger_words = item["trigger"]
                trigger_type = item['event_type']
                sentence_process = f"{trigger_words} <SEP> {trigger_words} <SEP> {trigger_words} <SEP> {sentence}" # 
                # 获取上下文嵌入（假设你的函数返回 numpy 数组）
                embedding = get_contextual_embedding(sentence, trigger_words, tokenizer, model, device, 512)
                if embedding is None:
                    continue
                embeddings.append(embedding.tolist())  # 转为 list
                metadatas.append({"sentence": sentence, "sentence_process": sentence_process, "word": trigger_words, "word_type": trigger_type})
                documents.append(trigger_words)
                ids.append(f"id_{idx}")  # 生成唯一 ID（可自定义）
                idx += 1
                
            
            for neg_sam_info in data["trigger_info"]["negative_triggers"]:
                neg_sam = neg_sam_info["trigger"]
                event_type = neg_sam_info["event_type"]
                sentence_process = f"{neg_sam} <SEP> {neg_sam} <SEP> {neg_sam} <SEP> {sentence}" # 
                embedding = get_contextual_embedding(sentence, neg_sam, tokenizer, model, device, 512)
                if embedding is None:
                    continue
                embeddings.append(embedding.tolist())  # 转为 list
                metadatas.append({"sentence": sentence, "sentence_process": sentence_process, "word": neg_sam, "word_type": event_type})
                documents.append(neg_sam)
                ids.append(f"id_{idx}")  # 生成唯一 ID（可自定义）
                idx += 1

        batch_size = 5000  # 确保小于报错中的限制（5461）
        for i in range(0, len(documents), batch_size):
            collection.add(
                embeddings=embeddings[i:i+batch_size],
                documents=documents[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size],
                ids=ids[i:i+batch_size] if ids else None
            )
    
    return collection  # 返回集合对象（替代原来的 db_word）


def create_db_sentence(train_data, tokenizer, model, device):
    # 初始化 ChromaDB 客户端
    client = PersistentClient()
    
    try:
        collection = client.get_collection(name="maven_sentence_bert_large")
    except:
        # 创建或获取集合（Collection）
        collection = client.create_collection(
            name="maven_sentence_bert_large",
            embedding_function=None,  # 禁用默认嵌入
            metadata={"hnsw:space": "cosine"}
        )
        
        embeddings = []
        metadatas = []
        documents = []
        ids = []  # ChromaDB 1.0+ 需要显式指定 ID（可选）

        idx = 0
        for data in tqdm(train_data, desc="create sentence_db"):
            sentence = data['sentence']
            for item in data['trigger_info']["triggers"]:
                trigger_words = item["trigger"]
                trigger_type = item['event_type']
                
                # 获取句子嵌入（假设你的函数返回 numpy 数组）
                embedding = get_sentence_embedding(sentence, tokenizer, model, device, 512)
                sentence_process = f"{trigger_words} <SEP> {trigger_words} <SEP> {trigger_words} <SEP> {sentence}" # 
                embeddings.append(embedding.tolist())  # 转为 list
                metadatas.append({"sentence": sentence, "sentence_process": sentence_process, "word": trigger_words, "word_type": trigger_type})
                documents.append(sentence)
                ids.append(f"id_{idx}")  # 生成唯一 ID（可自定义）
                idx += 1

        batch_size = 5000  # 确保小于报错中的限制（5461）
        for i in range(0, len(documents), batch_size):
            collection.add(
                embeddings=embeddings[i:i+batch_size],
                documents=documents[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size],
                ids=ids[i:i+batch_size] if ids else None
            )
        
    return collection  # 返回集合对象


if __name__ == "__main__":
    device = torch.device("cuda:1")
    model_name = "bert-large-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name, lowercase=True)
    model = AutoModel.from_pretrained(model_name).to(device)

    train_data_file_path = r'./maven_dataset/MAVEN/train_trigger_post_process.json'
    dev_data_file_path = r'./maven_dataset/MAVEN/valid_trigger_post_process.json'

    maven_data = []
    with open(train_data_file_path, 'r', encoding='utf-8') as f:
        maven_data.extend(json.load(f))
    with open(dev_data_file_path, 'r', encoding='utf-8') as f:
        maven_data.extend(json.load(f))
        
    '''
    client = PersistentClient('./chroma')
    client.delete_collection("maven_contextual_words_bert_large")
    client.delete_collection("maven_sentence_bert_large")
    '''
    create_db_word(maven_data, tokenizer, model, device)
    create_db_sentence(maven_data, tokenizer, model, device)