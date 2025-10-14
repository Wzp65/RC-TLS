import os
import json
from tqdm import tqdm
import argparse
import copy

import torch

from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig
from sentence_transformers import SentenceTransformer

from chromadb import PersistentClient
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from utils import get_contextual_embedding, get_sentence_embedding, safe_cosine, overlap_percentage, day_difference, completion_with_llm
from create_chroma_bert_large import create_db_word, create_db_sentence
from prompt_template import SAME_EVENT_JUDGMENT_PROMPT_TMP, SAME_EVENT_JUDGMENT_PROMPT


from openai import OpenAI
import openai
import requests

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyword", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default="./datasets")
    parser.add_argument("--maven_trigger_pos_path", type=str, default="./maven_dataset/MAVEN/trigger_pos_statistics.json")
    parser.add_argument("--model", type=str, default="bert-large-cased")
    parser.add_argument("--Qwen_model", type=str, default="/mnt/sdb1/wangzeping2023/Qwen/Qwen3-4B")
    args = parser.parse_args()
    return args


API_SECRET_KEY = "sk-zk2efaef69e398728b813df58e9b635c0b668d61f4f68d5d"
BASE_URL = "https://api.zhizengzeng.com/v1/"

# chat with other model
def chat_completions4(prompt_str, split_str):
    client = OpenAI(api_key=API_SECRET_KEY, base_url=BASE_URL)
    resp = client.chat.completions.create(
        model="qwen3-4b",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_str}
        ]
    )
    
    if resp and hasattr(resp, 'choices') and resp.choices:
        compressed_text = resp.choices[0].message.content
    else:
        print("Error: Invalid API response", resp)
        compressed_text = ""
    compressed_content = compressed_text.split(split_str)[-1].strip()
    
    first_part = compressed_content.split("#################", 1)[0]
    first_part = first_part.split("###", 1)[0].strip()
    
    return first_part


def find_sublist_index(words_text, trigger_list):
    n = len(trigger_list)
    for i in range(len(words_text) - n + 1):
        if words_text[i:i+n] == trigger_list:
            return i
    return -1


def match_trigger_syntax(words_text, words_pos, len_to_trigger_syntax, max_trigger_syntax_len):
    candidate_list = []
    for l in range(1, max_trigger_syntax_len + 1):
        l_trigger_syn_list = len_to_trigger_syntax[l]
        for t_s in l_trigger_syn_list:
            t_s_list = t_s.strip().split(" ")
            head_index = find_sublist_index(words_pos, t_s_list)
            if head_index != -1:
                c_trigger_pos = " ".join(words_pos[head_index: head_index + l])
                c_trigger_word = " ".join(words_text[head_index: head_index + l])
                candidate_list.append(c_trigger_word)
    return candidate_list


def extract_candidate_triggers(des_path, k_dataset_pos_path, trigger_syntax_list, k):
    max_trigger_syntax_len = 0
    len_to_trigger_syntax = dict()
    for trigger_syntax in trigger_syntax_list:
        trigger_syntax_len = len(trigger_syntax.split(" "))
        max_trigger_syntax_len = max(max_trigger_syntax_len, trigger_syntax_len)
        if trigger_syntax_len not in len_to_trigger_syntax.keys():
            len_to_trigger_syntax[trigger_syntax_len] = []
        len_to_trigger_syntax[trigger_syntax_len].append(trigger_syntax)

    with open(k_dataset_pos_path, "r", encoding='utf-8') as f:
        data_list = json.load(f)
    
    for d in tqdm(data_list, desc=f"{k} candidate trigger extractions"):
        sentence = d["content"]
        words_text = d["words_text"]
        words_pos = d["words_pos"]
        candidate_triggers =  match_trigger_syntax(words_text, words_pos, len_to_trigger_syntax, max_trigger_syntax_len)
        d["candidate_triggers"] = candidate_triggers
    
    with open(des_path, "w", encoding='utf-8') as f:
        data_list = json.dump(data_list, f, ensure_ascii=False, indent=4)


def retrieve_process(db_word, db_sentence, query_word, query_word_emb, query_sent, query_sent_emb, tokenizer, model, alpha, minim_score):
    # 检索相似单词
    if query_word_emb is None:
        return []
    results_words = db_word.query(
        query_embeddings=[query_word_emb.tolist()],
        n_results=4,
        include=["metadatas", "distances", "documents"]
    )
    if results_words is None:
        raise ValueError("Query failed: results_words is None")

    # 确保结果字段存在
    if not all(key in results_words for key in ["documents", "metadatas", "distances"]):
        raise ValueError("Missing expected keys in results_words")


    results_sentences = db_sentence.query(
        query_embeddings=[query_sent_emb.tolist()],
        n_results=2,
        include=["metadatas", "distances", "documents"]
    )

    sim_list = []
    sent_list = []
    id_ = 0
    for word, metadata, distance in zip(results_words["documents"][0], results_words["metadatas"][0], results_words["distances"][0]):
        sim_dic = dict()
        sim_dic['id'] = id_
        sim_dic['query_trigger'] = query_word
        sim_dic['word'] = word
        sim_dic['word_type'] = metadata['word_type']
        sim_dic['sentence'] = metadata['sentence']
        sent_list.append(sim_dic['sentence'])
        sim_dic['sentence_process'] = metadata['sentence_process']
        sim_dic['word_dis'] = 1.0 - distance
        if sim_dic['word_dis'] < minim_score:
            continue
        sim_dic['sent_dis'] = 0.0
        sim_dic['total_dis'] = 0.0
        sim_list.append(sim_dic)
        id_ += 1
    if len(sim_list) == 0:
        return None, None
    for sent_text, distance, metadata in zip(results_sentences["documents"][0], results_sentences["distances"][0], results_sentences["metadatas"][0]):
        if sent_text in sent_list:
            continue
        sim_dic = dict()
        sim_dic['id'] = id_
        sim_dic['query_trigger'] = query_word
        sim_dic['word'] = metadata['word']
        sim_dic['word_type'] = metadata['word_type']
        sim_dic['sentence'] = sent_text
        sim_dic['sentence_process'] = metadata['sentence_process']
        sim_dic['word_dis'] = 0.0
        sim_dic['sent_dis'] = 1.0 - distance
        sim_dic['total_dis'] = 0.0
        sim_list.append(sim_dic)
        id_ += 1
    
    for sim_dic in sim_list:
        if sim_dic['word_dis'] == 0.0:
            inputs = tokenizer(sim_dic['word'], return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs).last_hidden_state[0, 0, :].cpu().numpy()
            
            cos_sim = safe_cosine(outputs, query_word_emb)
            
            sim_dic['word_dis'] = cos_sim
            
        if sim_dic['sent_dis'] == 0.0:
            inputs = tokenizer(sim_dic['sentence'], return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs).last_hidden_state[0].mean(dim=0).cpu().numpy()
            
            cos_sim = safe_cosine(outputs, query_sent_emb)
            
            sim_dic['sent_dis'] = cos_sim
            
        
        sim_dic['total_dis'] = float(alpha * sim_dic['word_dis'] + (1.0 - alpha) * sim_dic['sent_dis'])
        #print(sim_dic['total_dis'])
    sorted_sim_list = sorted(sim_list, key=lambda x: x['total_dis'], reverse=True)
    
    return [sim_dic for sim_dic in sorted_sim_list if sim_dic['total_dis'] > minim_score]


def candidate_coarse_filtering(k_coarse_filter_path, k_candidate_path, model, tokenizer, db_word, db_sentence, device, k):
    with open(k_candidate_path, "r", encoding="utf-8") as f:
        candidate_info_lists = json.load(f)
    candidate_coarse_filter_list = []
    for cand_info in tqdm(candidate_info_lists, desc=f"coarse filtering candidates in {k}"):
        coarse_cand_dic = dict()
        cand_triggers = cand_info["candidate_triggers"]
        sentence = cand_info["content"]
        coarse_cand_dic["title"] = cand_info["title"]
        coarse_cand_dic["pubtime"] = cand_info["pubtime"]
        coarse_cand_dic["article_id"] = cand_info["article_id"]
        coarse_cand_dic["sent_index"] = cand_info["sent_index"]
        coarse_cand_dic["content"] = cand_info["content"]
        coarse_cand_dic["time"] = cand_info["time"]
        coarse_cands = []

        for pred in cand_triggers:
            query_sent = f"{pred} <SEP> {pred} <SEP> {pred} <SEP> {sentence}"
            query_word = pred
            query_sent_emb = get_sentence_embedding(query_sent, tokenizer, model, device, 512)
            query_word_emb = get_contextual_embedding(sentence, query_word, tokenizer, model, device, 512)
            sim_list = retrieve_process(db_word, db_sentence, query_word, query_word_emb, query_sent, query_sent_emb, tokenizer, model, 0.65, 0.65)
            if len(sim_list) == 0:
                continue
            if sim_list[0] is None:
                continue
            if sim_list[0]["word_type"] == "trigger_none":
                continue
            coarse_cands.append(pred)
        
        coarse_cand_dic["coarse_candidates"] = coarse_cands
        candidate_coarse_filter_list.append(coarse_cand_dic)
        
    with open(k_coarse_filter_path, "w", encoding="utf-8") as f:
        json.dump(candidate_coarse_filter_list, f, ensure_ascii=False, indent=4)


def is_preposition(word):
    """判断单个单词是否是介词"""
    prepositions = {
        'of', 'on', 'in', 'at', 'by', 'for', 'with', 'about', 'against',
        'between', 'into', 'through', 'during', 'before', 'after', 'above',
        'below', 'to', 'from', 'up', 'down', 'over', 'under', 'across',
        'along', 'around', 'behind', 'beneath', 'beside', 'beyond', 'near',
        'off', 'onto', 'out', 'outside', 'past', 'since', 'throughout',
        'toward', 'underneath', 'until', 'upon', 'within', 'without'
    }
    return word.lower() in prepositions


def remove_duplicate(k_coarse_filter_path, k_filter_path, keyword):
    with open(k_coarse_filter_path, "r", encoding='utf-8') as f:
        data_list = json.load(f)
    for d in tqdm(data_list, desc=f"remove {keyword} duplicate triggers"):
        candidates = d["coarse_candidates"]
        if len(candidates) == 0:
            continue
        new_cands = []
        new_cands.append(candidates[0])
        for c in range(1, len(candidates)):
            if len(new_cands) == 0:
                new_cands.append(candidates[c])
                continue
            new_cands_tmp = copy.deepcopy(new_cands)
            for n_c in new_cands_tmp:
                c_set = set(candidates[c].strip().split(" "))
                n_c_set = set(n_c.strip().split(" "))
                common_set = c_set & n_c_set
                if len(common_set) == 0:
                    new_cands.append(candidates[c])
                    continue
                if len(common_set) == 1:
                    common_word = list(common_set)[0]
                    if is_preposition(common_word):
                        if common_word in new_cands:
                            new_cands.remove(common_word)
                        continue
                
                common = " ".join(list(common_set))
                if n_c in new_cands:
                    new_cands.remove(n_c)
                if common in new_cands:
                    new_cands.remove(common)
                new_cands.append(common)
        
        new_cands_tmp = copy.deepcopy(new_cands)
        one_set = set()
        overlap_set = set()
        for n in new_cands_tmp:
            if is_preposition(n):
                if n in new_cands:
                    new_cands.remove(n)
            n_list = n.split(" ")
            for n_w in n_list:
                if n_w not in one_set:
                    one_set.add(n_w)
                else:
                    if not is_preposition(n_w):
                        if n in new_cands:
                            new_cands.remove(n)

        d["coarse_candidates"] = new_cands

    with open(k_filter_path, "w", encoding='utf-8') as f:
        json.dump(data_list, f, ensure_ascii=False, indent=4)


def create_gte_chroma(k_filter_path, dataset, keyword, gte_model):
    # 初始化 ChromaDB 客户端
    client = PersistentClient()
    
    try:
        collection = client.get_collection(name=f"gte_{dataset}_{keyword}")
    except:
        # 创建或获取集合（Collection）
        collection = client.create_collection(
            name=f"gte_{dataset}_{keyword}",
            embedding_function=None,  # 禁用默认嵌入
            metadata={"hnsw:space": "cosine"}
        )
        
        with open(k_filter_path, "r", encoding='utf-8') as f:
            train_data = json.load(f)

        embeddings = []
        metadatas = []
        documents = []
        ids = []  # ChromaDB 1.0+ 需要显式指定 ID（可选）

        idx = 0
        for data in tqdm(train_data, desc=f"create gte_{dataset}_{keyword}"):
            title = data["title"]
            if title is None:
                title = ""
            pubtime = data["pubtime"]
            article_id = data["article_id"]
            sent_index = data["sent_index"]
            content = data["content"]
            time = data["time"]
            if len(data["coarse_candidates"]) == 0:
                triggers = ""
            else:
                triggers = ",".join(data["coarse_candidates"])
            
            # 获取句子嵌入（假设你的函数返回 numpy 数组）
            embedding = gte_model.encode(content)
                
            embeddings.append(embedding.tolist())  # 转为 list
            metadatas.append({"title": title, "pubtime": pubtime, "article_id": article_id, "sent_index": sent_index, "time": time, "triggers": triggers, "event_id": idx})
            documents.append(content)
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


def retrieve_gte(db_gte_sent, query_sent, query_sent_emb, gte_model):
    
    results = db_gte_sent.query(
        query_embeddings=[query_sent_emb.tolist()],
        n_results=12,
        include=["metadatas", "distances", "documents"]
    )

    sim_list = []
    for content, metadata, distance in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        sim_dic = dict()
        sim_dic['content'] = content
        sim_dic["triggers"] = metadata['triggers'] 
        sim_dic['title'] = metadata['title']
        sim_dic['pubtime'] = metadata['pubtime']
        sim_dic['article_id'] = metadata['article_id']
        sim_dic['sent_index'] = metadata['sent_index']
        sim_dic['time'] = metadata['time'] 
        sim_dic["event_id"] = metadata["event_id"]
        sim_list.append(sim_dic)
    
    return sim_list


def same_events_finding(des_dir, k_filter_path, gte_model, db_gte_sent, dataset, keyword, prompt, Qwen_tokenizer, Qwen_model):
    same_event_list = []
    with open(k_filter_path, "r", encoding="utf-8") as f:
        data_list = json.load(f)
    cos_sim_file_path = os.path.join(des_dir, "cos_sim_pair.json")
    same_event_file_path = os.path.join(des_dir, "same_event.json")
    if os.path.exists(os.path.join(des_dir, "same_event.jsonl")):
        with open(os.path.join(des_dir, "same_event.jsonl"), "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                same_event_list.append(data)
    for d in tqdm(data_list, desc=f"same events finding:{dataset}-{keyword}"):
        content = d["content"]
        article_id = d["article_id"]
        event_time = d["time"]
        event_pubtime = d["pubtime"]
        if len(d["coarse_candidates"]) == 0:
            continue
        content_emb = gte_model.encode(content)
        re_data = retrieve_gte(db_gte_sent, content, content_emb, gte_model)
        
        event_id_list = [re_c["event_id"] for re_c in re_data if (re_c["content"] == content and re_c["article_id"] == article_id)]
        if len(event_id_list) == 0:
            continue
        event_id = event_id_list[0]

        flag = 0
        for same_event_info in same_event_list:
            if same_event_info[0][1] == event_id:
                flag = 1
                break
        
        if flag == 1:
            continue

        re_data = [re_c for re_c in re_data if ((re_c["content"] != content and re_c["article_id"] == article_id) or re_c["article_id"] != article_id)]
        re_data = [re_c for re_c in re_data if (day_difference(re_c["pubtime"], event_pubtime) <= 7 or day_difference(re_c["time"], event_time) <= 7 or overlap_percentage(content, re_c["content"], 40))]
        re_content = [re_c["content"] for re_c in re_data]
        
        with open(cos_sim_file_path, "a", encoding='utf-8') as f:
            json.dump([content, re_content], f, ensure_ascii=False, indent=4)
            f.write("\n")

        same_event = [[content, event_id], []]

        for re_d in re_data:
            query_1 = content
            query_2 = re_d["content"]
            keywords_1 = ",".join(d["coarse_candidates"])
            keywords_2 = re_d["triggers"]
            current_prompt = prompt.format(input_1=query_1, input_2=query_2, keywords_1=keywords_1, keywords_2=keywords_2)
            split_str = f"### The determination of whether the above two statements describing the same event is"
            res = completion_with_llm(Qwen_tokenizer, Qwen_model, current_prompt, split_str, temperature=0.0, stop_tokens=["#################", "###"], max_len=2)
            # res = chat_completions4(current_prompt, split_str)
            if "yes" in res.lower():
                neighbor_event_id = re_d["event_id"]
                same_event[1].append([query_2, neighbor_event_id])
                         
        with open(os.path.join(des_dir, "same_event.jsonl"), "a", encoding='utf-8') as f:
            f.write(json.dumps(same_event, ensure_ascii=False) + "\n")

        same_event_list.append(same_event)

    with open(same_event_file_path, "w", encoding='utf-8') as f:
        json.dump(same_event_list, f, ensure_ascii=False, indent=4)


def double_retreive(content_emb, query_event, query_event_id, query_article_id, query_pubtime, db_gte_sent, gte_model):
    re_data = retrieve_gte(db_gte_sent, query_event, content_emb, gte_model)
    re_data = [re_c for re_c in re_data if re_c["article_id"] == query_article_id and query_event_id != re_c["event_id"] and re_c["pubtime"] == query_pubtime and re_c["time"] == query_pubtime]
    re_data_tmp = copy.deepcopy(re_data)
    for re_d in re_data_tmp:
        query_event = re_d["content"]
        query_event_id_1 = re_d["event_id"]
        content_emb = gte_model.encode(query_event)
        re_data_add = retrieve_gte(db_gte_sent, query_event, content_emb, gte_model)
        re_data_add = [re_c for re_c in re_data_add if re_c["article_id"] == query_article_id and query_event_id != re_c["event_id"] and query_event_id_1 != re_c["event_id"] and re_c["pubtime"] == query_pubtime and re_c["time"] == query_pubtime]
        re_data.extend(re_data_add)

    re_data_remove_duplicate = []
    data_id = set()
    for re_d in re_data:
        event_id = re_d["event_id"]
        if event_id in data_id:
            continue
        data_id.add(event_id)
        re_data_remove_duplicate.append(re_d)
    re_event_and_id = [[e["content"], e["event_id"]] for e in re_data_remove_duplicate]
    return re_event_and_id, re_data_remove_duplicate


def acquire_connected_event(des_dir, same_event_file_path, gte_model, db_gte_sent, dataset, keyword):
    with open(same_event_file_path, "r", encoding='utf-8') as f:
        same_event_list = json.load(f)
    
    same_connected_event_file_path = os.path.join(des_dir, "same_and_relation_events.json")
    same_relation_events_list = []
    retro_event = dict()
    for same_event_info in tqdm(same_event_list, desc=f"find relational events in same article:{dataset}-{keyword}"):
        query_event = same_event_info[0][0]
        query_event_id = same_event_info[0][1]
        
        results = db_gte_sent.get(
            where={"event_id": query_event_id}  # 元数据过滤条件
        )
        
        query_article_id_1 = results["metadatas"][0]["article_id"]

        query_event_time = results["metadatas"][0]["time"]
        query_pubtime = results["metadatas"][0]["pubtime"]

        # 必须与本文主要内容相关
        if query_pubtime != query_event_time:
            if query_event_id not in retro_event.keys():
                retro_event[query_event_id] = query_event
            continue
        
        content_emb = gte_model.encode(query_event)
        re_event_and_id, _ = double_retreive(content_emb, query_event, query_event_id, query_article_id_1, query_pubtime, db_gte_sent, gte_model)
        same_relation_events = [[query_event, query_event_id, re_event_and_id], []]

        same_article_events = dict()
        same_event_infor_tmp = copy.deepcopy(same_event_info[1])
        for iden_event_info in same_event_infor_tmp:
            query_event = iden_event_info[0]
            query_event_id = iden_event_info[1]
            results = db_gte_sent.get(
                where={"event_id": query_event_id}  # 元数据过滤条件
            )
            query_article_id = results["metadatas"][0]["article_id"]
            if query_article_id not in same_article_events.keys():
                same_article_events[query_article_id] = []
            same_article_events[query_article_id].append(iden_event_info)

        for query_article_id, item in same_article_events.items():
            if query_article_id == query_article_id_1:
                continue
            
            re_event_and_id = []
            for iden_event_info in item:
                
                query_event = iden_event_info[0]
                query_event_id = iden_event_info[1]
                results = db_gte_sent.get(
                    where={"event_id": query_event_id}  # 元数据过滤条件
                )
                query_event_time = results["metadatas"][0]["time"]
                query_pubtime = results["metadatas"][0]["pubtime"]

                if query_pubtime != query_event_time:
                    if query_event_id not in retro_event.keys():
                        retro_event[query_event_id] = query_event
                    continue
        
                content_emb = gte_model.encode(query_event)
                re_event_and_id_tmp, _ = double_retreive(content_emb, query_event, query_event_id, query_article_id, query_pubtime, db_gte_sent, gte_model)
                re_event_and_id.extend(re_event_and_id_tmp)
                id_list = [re[1] for re in re_event_and_id]

                # 删除same_events中与检索到的内容相同的语句事件
                for info in same_event_infor_tmp:
                    if info[1] in id_list:
                        if info in same_event_info[1]:
                            same_event_info[1].remove(info)
                            break

            id_list = []
            re_event_and_id_tmp = copy.deepcopy(re_event_and_id)
            # 去除重复元素
            for info in re_event_and_id_tmp:
                if info[1] not in id_list:
                    id_list.append(info[1])
                else:
                    if info in re_event_and_id:
                        re_event_and_id.remove(info)
                
            same_relation_events_tmp = [query_event, query_event_id, re_event_and_id]
            same_relation_events[1].append(same_relation_events_tmp)
            
        same_relation_events_list.append(same_relation_events)
        
        with open(same_connected_event_file_path, "a", encoding="utf-8") as f:
            json.dump(same_relation_events, f, ensure_ascii=False, indent=4)
        
    with open(same_connected_event_file_path, "w", encoding="utf-8") as f:
        json.dump(same_relation_events_list, f, ensure_ascii=False, indent=4)
    
    with open(os.path.join(des_dir, "retrospect_events.json"), "w", encoding="utf-8") as f:
        json.dump(retro_event, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    args = get_argparser()
    des_root_dir = f"./processing/{args.dataset}"
    dataset_dir = f"{args.dataset_path}/{args.dataset}"

    device = torch.device("cuda:2")
    '''
    model_name = args.model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    '''
    '''
    Qwen_tokenizer = AutoTokenizer.from_pretrained(args.Qwen_model)
    Qwen_model = AutoModelForCausalLM.from_pretrained(args.Qwen_model, device_map="cuda:2")
    '''
    trigger_syntax_list = []
    with open(args.maven_trigger_pos_path, "r", encoding="utf-8") as f:
        st_dic = json.load(f)
        for pos_type in st_dic.keys():
            trigger_syntax_list.append(pos_type)

    if args.keyword == "all":
        keyword = [name for name in os.listdir(dataset_dir) 
                if os.path.isdir(os.path.join(dataset_dir, name))]
    else:
        keyword = args.keyword.split(',')
    
    for k in tqdm(keyword, desc="event detection"):
        des_dir = os.path.join(des_root_dir, k)
        os.makedirs(des_dir, exist_ok=True)
        k_candidate_path = os.path.join(des_dir, "candidate_triggers.json")
        k_dataset_pos_path = os.path.join(dataset_dir, k, "preprocess_to_sent.json")

        # extract_candidate_triggers(k_candidate_path, k_dataset_pos_path, trigger_syntax_list, k)
        k_coarse_filter_path = os.path.join(des_dir, "coarse_candidate_filtering.json")
        '''
        train_data = []
        db_word = create_db_word(train_data, tokenizer, model, device)
        db_sentence = create_db_sentence(train_data, tokenizer, model, device)
        del train_data
        '''
        #candidate_coarse_filtering(k_coarse_filter_path, k_candidate_path, model, tokenizer, db_word, db_sentence, device, k)
        k_filter_path = os.path.join(des_dir, "candidate_filtering.json")
        #remove_duplicate(k_coarse_filter_path, k_filter_path, k)
        
        gte_model = SentenceTransformer('thenlper/gte-large').to(device)
        db_gte_sent = create_gte_chroma(k_filter_path, args.dataset, k, gte_model)

        if args.dataset == "t17":
            if k != "syria":
                prompt = SAME_EVENT_JUDGMENT_PROMPT
            else:
                prompt = SAME_EVENT_JUDGMENT_PROMPT_TMP
        same_events_finding(des_dir, k_filter_path, gte_model, db_gte_sent, args.dataset, k, prompt, Qwen_tokenizer, Qwen_model)
        
        same_event_file_path = os.path.join(des_dir, "same_event.json")
        acquire_connected_event(des_dir, same_event_file_path, gte_model, db_gte_sent, args.dataset, k)  