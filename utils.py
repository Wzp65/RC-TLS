import os
import json
import numpy as np
import torch

from datetime import datetime, timezone

from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList

def safe_cosine(v1, v2, eps=1e-8):
    if v1 is None or v2 is None:
        return 0.0
    # 计算范数时直接防止除零
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    # 处理零向量情况
    if norm1 < eps or norm2 < eps:
        return 0.0  # 或根据业务需求返回其他默认值
    
    # 标准余弦相似度计算
    return np.dot(v1, v2) / (norm1 * norm2)


def get_sentence_embedding(sentence, tokenizer, model, device, max_length):
    inputs = tokenizer(sentence.lower(), return_tensors="pt", truncation=True, max_length=max_length).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state[0].mean(dim=0).cpu().numpy()  # [embed_dim]
    return embedding


def get_contextual_embedding(sentence, target_word, tokenizer, model, device, max_length):
    #print(target_word.lower())
    page_content = f"{target_word} <SEP> {target_word} <SEP> {target_word} <SEP> {sentence}" # 
    #print(page_content.lower())
    inputs = tokenizer(page_content.lower(), return_tensors="pt", truncation=True, max_length=max_length).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[0]  # [seq_len, embed_dim]
    
    # 找到目标单词的位置
    word_tokens = tokenizer.tokenize(page_content.lower(), return_tensors="pt", truncation=True, max_length=max_length)
    #print(word_tokens)
    word_pieces = tokenizer.tokenize(target_word.lower(), return_tensors="pt", truncation=True, max_length=max_length)
    #print(word_pieces)
    try:
        start_idx = find_word_position(word_tokens, word_pieces)
    except ValueError:
        return None
    return embeddings[start_idx:start_idx+len(word_pieces)].mean(dim=0).cpu().numpy()


def find_word_position(full_tokens, word_tokens):
    """定位单词在分词结果中的起始位置"""
    pos_idx = 0
    for i in range(len(full_tokens) - len(word_tokens) + 1):
        
        if full_tokens[i:i+len(word_tokens)] == word_tokens:
            pos_idx += 1
            if pos_idx == 4:
                #print(i)
                return i + 1  # +1跳过[CLS]
    raise ValueError(f"Word not found: {word_tokens}")


def overlap_percentage(str1, str2, percentage):
    # 找到较短的单词事件语句
    list_1 = str1.split(' ')
    list_2 = str2.split(' ')
    short_list = min(list_1, list_2, key=len)
    long_list = max(list_2, list_1, key=len)
    
    # 计算重叠部分的长度
    overlap_length = 0
    for i in range(len(short_list)):
        for j in range(len(long_list)):
            if short_list[i] == long_list[j]:
                overlap_length += 1
                break

    # 计算重叠部分占较短的百分比
    percentage1 = (overlap_length / len(short_list)) * 100
    
    # 判断是否大于70%
    return percentage1 > percentage


def parse_time(time_str):
    # 尝试解析带时区的时间
    try:
        return datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S%z")
    except ValueError:
        # 如果不带时区，则视为本地时间（或指定默认时区）
        return datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)  # 假设UTC

def to_utc(dt):
    return dt.astimezone(timezone.utc)

def day_difference(time_str1, time_str2):
    dt1 = to_utc(parse_time(time_str1))
    dt2 = to_utc(parse_time(time_str2))
    return abs((dt2 - dt1).days)


class StopAtSpecificTokenCriteria(StoppingCriteria):
    def __init__(self, stop_tokens: list, tokenizer):
        # 将停止标记转换为它们对应的 token ID
        self.stop_token_ids = [tokenizer.encode(stop_token, return_tensors='pt').squeeze(0).tolist() for stop_token in stop_tokens]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # 检查生成的 token 是否包含停止标记
        for stop_token_ids in self.stop_token_ids:
            # 检查生成序列是否以指定的 token 结尾
            if input_ids[0, -len(stop_token_ids):].tolist() == stop_token_ids:
                return True
        return False


def completion_with_llm(tokenizer, model, prompt_str, split_str, temperature=0.0, stop_tokens=[], max_len=10):
    model_input = tokenizer(prompt_str, return_tensors="pt").to(model.device)
    stop_criteria = StopAtSpecificTokenCriteria(stop_tokens=stop_tokens, tokenizer=tokenizer)
    
    model.eval()
    with torch.no_grad():
        if temperature != 0.0:
            completion = model.generate(
                **model_input,
                max_new_tokens=max_len,
                stopping_criteria=StoppingCriteriaList([stop_criteria]),
                temperature=temperature,
                do_sample=True,
                top_p=0.2,
                top_k=2,
                #pad_token_id=model.config.eos_token_id
            )[0]
        else:
            completion = model.generate(
                **model_input,
                max_new_tokens=max_len,
                stopping_criteria=StoppingCriteriaList([stop_criteria]),
                do_sample=False
                #pad_token_id=model.config.eos_token_id
            )[0]
    compressed_text = tokenizer.decode(completion, skip_special_tokens=True)
    compressed_content = compressed_text.split(split_str)[-1].strip()
    
    first_part = compressed_content.split("#################", 1)[0]
    first_part = first_part.split("###", 1)[0].strip()
    
    return first_part