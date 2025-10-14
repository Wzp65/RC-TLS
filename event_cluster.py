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
import chromadb

from utils import completion_with_llm
from create_chroma_bert_large import create_db_word, create_db_sentence
from prompt_template import RELATION_STATEMENTS_SUMMARY_PROMPT, RELATION_STATEMENTS_SUMMARY_PROMPT_TMP, RELATION_CLUSTER_SPLIT_PROMPT, RELATION_CLUSTER_SPLIT_PROMPT_TMP, DAY_SUMMARIZE_PROMPT_TMP, DAY_SUMMARIZE_PROMPT


from openai import OpenAI
import openai
import requests


os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyword", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default="./datasets")
    parser.add_argument("--model", type=str, default="bert-large-cased")
    parser.add_argument("--Qwen_model", type=str, default="/mnt/sdb1/wangzeping2023/Qwen/Qwen3-4B")
    args = parser.parse_args()
    return args


API_SECRET_KEY = "sk-zk2645621afc0eb3792dca090f859038b1268e723c7b4c39"
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


def relation_events_summarization(dir_path, dataset, keyword, prompt):
    same_and_relation_events_file = os.path.join(dir_path, "same_and_relation_events.json")
    with open(same_and_relation_events_file, "r", encoding='utf-8') as f:
        same_and_relation_events = json.load(f)

    retrospect_events_file = os.path.join(dir_path, "retrospect_events.json")
    with open(retrospect_events_file, "r", encoding="utf-8") as f:
        retrospect_events = json.load(f)

    retrospect_events = {int(key): value for key, value in retrospect_events.items()}

    same_and_relation_events_summaries_list = []
    multievents_group = dict()
    for same_and_relation_event in tqdm(same_and_relation_events, desc=f"relational events summarization:{dataset}-{keyword}"):
        if len(same_and_relation_event[0][2]) == 0:
            if same_and_relation_event[0][1] not in multievents_group.keys():
                multievents_group[same_and_relation_event[0][1]] = same_and_relation_event[0][0]
            else:
                multievents_group[same_and_relation_event[0][1]] = same_and_relation_event[0][0] if len(same_and_relation_event[0][0]) > len(multievents_group[same_and_relation_event[0][1]]) else multievents_group[same_and_relation_event[0][1]]
            same_and_relation_events_summaries = [[same_and_relation_event[0][0], same_and_relation_event[0][1]], []]
        else:
            relation_set = set()
            relation_set.add(same_and_relation_event[0][1])
            content_list = []
            for relation_event_info in same_and_relation_event[0][2]:
                relation_set.add(relation_event_info[1])
                content_list.append(relation_event_info[0])
            relation_tuple = tuple(sorted(list(relation_set)))

            if relation_tuple not in multievents_group.keys():
                content = "\n".join(content_list)
                current_prompt = prompt.format(Input_Set=content)
                split_str = "### The Summarization of the above Input Set 3 is"
                res = chat_completions4(current_prompt, split_str)
                multievents_group[relation_tuple] = res
                same_and_relation_events_summaries = [[res, same_and_relation_event[0][1]], []]
            else:
                same_and_relation_events_summaries = [[multievents_group[relation_tuple], same_and_relation_event[0][1]], []]
                if relation_tuple[0] in retrospect_events.keys():
                    del retrospect_events[relation_tuple[0]]
        
        for same_events in same_and_relation_event[1]:
            if len(same_events[2]) == 0:
                if same_events[1] not in multievents_group.keys():
                    multievents_group[same_events[1]] = same_events[0]
                else:
                    multievents_group[same_events[1]] = same_events[0] if len(same_events[0]) > len(multievents_group[same_events[1]]) else multievents_group[same_events[1]]
                same_and_relation_events_summaries[1].append([same_events[0], same_events[1]])
            else:
                relation_set = set()
                relation_set.add(same_events[1])
                content_list = []
                for relation_event_info in same_events[2]:
                    relation_set.add(relation_event_info[1])
                    content_list.append(relation_event_info[0])
                relation_tuple = tuple(sorted(list(relation_set)))

                if relation_tuple not in multievents_group.keys():
                    content = "\n".join(content_list)
                    current_prompt = prompt.format(Input_Set=content)
                    split_str = "### The Summarization of the above Input Set 3 is"
                    res = chat_completions4(current_prompt, split_str)
                    multievents_group[relation_tuple] = res
                    same_and_relation_events_summaries[1].append([res, same_events[1]])
                else:
                    same_and_relation_events_summaries[1].append([multievents_group[relation_tuple], same_events[1]])
                    if relation_tuple[0] in retrospect_events.keys():
                        del retrospect_events[relation_tuple[0]]
        same_and_relation_events_summaries_list.append(same_and_relation_events_summaries) 

        with open(os.path.join(dir_path, "same_events_summaries_1.json"), "a", encoding="utf-8") as f:
            json.dump(same_and_relation_events_summaries, f, ensure_ascii=False, indent=4)
            f.write("\n")
    
    same_and_relation_events_summaries_list = sorted(same_and_relation_events_summaries_list, key=lambda x:x[0][1])
    with open(os.path.join(dir_path, "same_events_summaries_1.json"), "w", encoding="utf-8") as f:
        json.dump(same_and_relation_events_summaries_list, f, ensure_ascii=False, indent=4)

    with open(retrospect_events_file, "w", encoding="utf-8") as f:
        json.dump(retrospect_events, f, ensure_ascii=False, indent=4)


def id2events_construction(dir_path, dataset, k):
    with open(os.path.join(dir_path, "same_events_summaries_1.json"), "r", encoding="utf-8") as f:
        same_events_summaries_list = json.load(f)
    with open(os.path.join(dir_path, "same_event.json"), "r", encoding="utf-8") as f:
        same_events_list = json.load(f)
    with open(os.path.join(dir_path, "same_and_relation_events.json"), "r", encoding="utf-8") as f:
        same_and_relation_events_list = json.load(f)
    with open(os.path.join(dir_path, "retrospect_events.json"), "r", encoding="utf-8") as f:
        retrospect_events = json.load(f)

    same_events_summaries_list = sorted(same_events_summaries_list, key=lambda x:x[0][1])
    same_events_list = sorted(same_events_list, key=lambda x:x[0][1])
    same_and_relation_events_list = sorted(same_and_relation_events_list, key=lambda x:x[0][1])
    
    retrospect_events = {int(key): value for key, value in retrospect_events.items()}

    assert len(same_events_summaries_list) == len(same_and_relation_events_list)
    id2events = dict()

    for same_events_summaries, same_and_relation_events in zip(tqdm(same_events_summaries_list, desc=f"id2event construction:{dataset}-{k}"), same_and_relation_events_list):
        assert same_events_summaries[0][1] == same_and_relation_events[0][1]
        event_id = same_and_relation_events[0][1]

        if len(same_and_relation_events[0][2]) == 0:
            if event_id in id2events.keys():
                id2events[event_id] = same_events_summaries[0][0] if len(same_events_summaries[0][0]) > len(id2events[event_id]) else same_events_summaries[0][0]
            else:
                id2events[event_id] = same_events_summaries[0][0]
        else:
            event_ids_set = set()
            event_ids_set.add(event_id)
            for relation_events_info in same_and_relation_events[0][2]:
                event_ids_set.add(relation_events_info[1])
                event_ids_tuple = tuple(sorted(list(event_ids_set)))
                if event_ids_tuple[0] in id2events.keys():
                    id2events[event_ids_tuple[0]] = same_events_summaries[0][0] if len(same_events_summaries[0][0]) > len(id2events[event_ids_tuple[0]]) else id2events[event_ids_tuple[0]]
                else:
                    id2events[event_ids_tuple[0]] = same_events_summaries[0][0]

        assert len(same_events_summaries[1]) == len(same_and_relation_events[1])
        for same_events_summary, same_events_info in zip(same_events_summaries[1], same_and_relation_events[1]):
            assert same_events_summary[1] == same_events_info[1]
            event_id = same_events_summary[1]
            if len(same_events_info[2]) == 0:
                if event_id in id2events.keys():
                    id2events[event_id] = same_events_summary[0] if len(same_events_summary[0]) > len(id2events[event_id]) else same_events_summary[0]
                else:
                    id2events[event_id] = same_events_summary[0]

            else:
                event_ids_set = set()
                event_ids_set.add(event_id)
                for relation_events_info in same_events_info[2]:
                    event_ids_set.add(relation_events_info[1])
                    event_ids_tuple = tuple(sorted(list(event_ids_set)))
                    if event_ids_tuple[0] in id2events.keys():
                        id2events[event_ids_tuple[0]] = same_events_summary[0] if len(same_events_summary[0]) > len(id2events[event_ids_tuple[0]]) else id2events[event_ids_tuple[0]]
                    else:
                        id2events[event_ids_tuple[0]] = same_events_summary[0]

    for key, value in retrospect_events.items():
        if key not in id2events.keys():
            id2events[key] = value

    with open(os.path.join(dir_path, "id2event.json"), "w", encoding="utf-8") as f:
        json.dump(id2events, f, ensure_ascii=False, indent=4)


def relation_events_clustering(dir_path, dataset, keyword):
    with open(os.path.join(dir_path, "same_event.json"), "r", encoding="utf-8") as f:
        same_events_list = json.load(f)
    with open(os.path.join(dir_path, "same_and_relation_events.json"), "r", encoding="utf-8") as f:
        same_and_relation_events_list = json.load(f)
    with open(os.path.join(dir_path, "id2event.json"), "r", encoding="utf-8") as f:
        id2event = json.load(f)
    
    id2event = {int(key): value for key, value in id2event.items()}

    event_pool = {}
    event2cluster = {}

    for same_and_relation_events in tqdm(same_and_relation_events_list, desc=f"relation clustering:{dataset}-{keyword}_1"):
        event_id = same_and_relation_events[0][1]

        if event_id not in event2cluster.keys():
            if len(event_pool) == 0:
                max_pool_len = 0
            else:
                max_pool_len = max(list(event_pool.keys())) + 1
            
            event_pool[max_pool_len] = [event_id]
            event2cluster[event_id] = max_pool_len
        
        cls_id = event2cluster[event_id]
        for relation_events_info in same_and_relation_events[0][2]:
            neibor_id = relation_events_info[1]
            if neibor_id not in event2cluster.keys():
                event_pool[cls_id].append(neibor_id)
                event2cluster[neibor_id] = cls_id
            else:
                neibo_cls_id = event2cluster[neibor_id]
                if neibo_cls_id == cls_id:
                    continue
                for n_id in event_pool[neibo_cls_id]:
                    if n_id not in event_pool[cls_id]:
                        event_pool[cls_id].append(n_id)
                    event2cluster[n_id] = cls_id
                del event_pool[neibo_cls_id]
        
        for same_event_info in same_and_relation_events[1]:
            same_event_id = same_event_info[1]
            if same_event_id not in event2cluster.keys():
                event_pool[cls_id].append(same_event_id)
                event2cluster[same_event_id] = cls_id
            else:
                same_cls_id = event2cluster[same_event_id]
                if same_cls_id == cls_id:
                    continue
                for n_id in event_pool[same_cls_id]:
                    if n_id not in event_pool[cls_id]:
                        event_pool[cls_id].append(n_id)
                    event2cluster[n_id] = cls_id
                del event_pool[same_cls_id]

            for relation_events_info in same_event_info[2]:
                neibor_id = relation_events_info[1]
                if neibor_id not in event2cluster.keys():
                    event_pool[cls_id].append(neibor_id)
                    event2cluster[neibor_id] = cls_id
                else:
                    neibo_cls_id = event2cluster[neibor_id]
                    if neibo_cls_id == cls_id:
                        continue
                    for n_id in event_pool[neibo_cls_id]:
                        if n_id not in event_pool[cls_id]:
                            event_pool[cls_id].append(n_id)
                        event2cluster[n_id] = cls_id
                    del event_pool[neibo_cls_id]
    
    for same_events_info in tqdm(same_events_list, desc=f"relation clustering:{dataset}-{keyword}_2"):
        event_id = same_events_info[0][1]
        if event_id not in event2cluster.keys():
            if len(event_pool) == 0:
                max_pool_len = 0
            else:
                max_pool_len = max(list(event_pool.keys())) + 1
            
            event_pool[max_pool_len] = [event_id]
            event2cluster[event_id] = max_pool_len
        
        cls_id = event2cluster[event_id]
        for relation_events_info in same_events_info[1]:
            neibor_id = relation_events_info[1]
            if neibor_id not in event2cluster.keys():
                event_pool[cls_id].append(neibor_id)
                event2cluster[neibor_id] = cls_id
            else:
                neibo_cls_id = event2cluster[neibor_id]
                if neibo_cls_id == cls_id:
                    continue
                for n_id in event_pool[neibo_cls_id]:
                    if n_id not in event_pool[cls_id]:
                        event_pool[cls_id].append(n_id)
                    event2cluster[n_id] = cls_id
                del event_pool[neibo_cls_id]

    event_pool_tmp = {}
    event2cluster_tmp = {}
    for key, value in event_pool.items():
        event_pool_tmp[key] = []
        for event_id in value:
            if event_id in id2event.keys():
                event_pool_tmp[key].append(event_id)
                event2cluster_tmp[event_id] = key
    
    with open(os.path.join(dir_path, "event_pool_relation.json"), "w", encoding="utf-8") as f:
        json.dump(event_pool_tmp, f, ensure_ascii=False, indent=4)
    with open(os.path.join(dir_path, "event2cluster_relation.json"), "w", encoding="utf-8") as f:
        json.dump(event2cluster_tmp, f, ensure_ascii=False, indent=4)


def cluster_splitting(cls_id, event_id_group_dic, event_pool, event2cluster):
    for event_id_list in event_id_group_dic.values():
        max_key = max(list(event_pool.keys())) if event_pool else 0
        unused_keys = [c_id for c_id in range(max_key) if c_id not in event_pool]
        new_keys = list(range(max_key + 1, max_key * 10))
        new_cls_id = min(unused_keys + new_keys)
        event_pool[new_cls_id] = event_id_list
        for event_id in event_id_list:
            event2cluster[event_id] = new_cls_id
    
    del event_pool[cls_id]


def relation_cluster_splitting(dir_path, dataset, keyword, device):
    '''
    Qwen_tokenizer = AutoTokenizer.from_pretrained(args.Qwen_model)
    Qwen_model = AutoModelForCausalLM.from_pretrained(args.Qwen_model, device_map="cuda:1")
    '''

    gte_model = SentenceTransformer('thenlper/gte-large').to(device)

    if dataset == "t17":
        if k != "mj":
            prompt = RELATION_CLUSTER_SPLIT_PROMPT_TMP
        else:
            prompt = RELATION_CLUSTER_SPLIT_PROMPT

    split_str = "### The determination of whether the above two statements describing the same event is"

    with open(os.path.join(dir_path, "event_pool_relation.json"), "r", encoding="utf-8") as f:
        event_pool = json.load(f)
    
    with open(os.path.join(dir_path, "event2cluster_relation.json"), "r", encoding="utf-8") as f:
        event2cluster = json.load(f)

    with open(os.path.join(dir_path, "id2event.json"), "r", encoding="utf-8") as f:
        id2event = json.load(f)
    
    event_pool = {int(key): value for key, value in event_pool.items()}
    event2cluster = {int(key): value for key, value in event2cluster.items()}
    id2event = {int(key): value for key, value in id2event.items()}
    
    relation_event_pairs = []

    event_pool_tmp = copy.deepcopy(event_pool)
    for cls_id, event_id_list in tqdm(event_pool_tmp.items(), desc=f"split relation clustering:{dataset}-{keyword}"):
        if len(event_id_list) > 1:
            
            event_id_group_dic = dict()
            flag_set = set()

            event_id_list_len = len(event_id_list)
            for e_idx_1 in range(15):
                if e_idx_1 < event_id_list_len:
                    event_id = event_id_list[e_idx_1]
                    content_1 = id2event[event_id]
                    if event_id in flag_set:
                        continue
                    flag_set.add(event_id)
                    event_id_group = [event_id]
                    event_id_group_dic[event_id] = event_id_group
                    rep_id = event_id
                    rep_content = ""
                    for e_idx_2 in range(e_idx_1 + 1, 15):
                        if e_idx_2 >= event_id_list_len:
                            break
                        event_id_tmp = event_id_list[e_idx_2]
                        content_2 = id2event[event_id_tmp]
                        if event_id_tmp in flag_set:
                            continue
                        current_prompt = prompt.format(input_1=content_1, input_2=content_2)
                        res = chat_completions4(current_prompt, split_str)
                        # res = completion_with_llm(Qwen_tokenizer, Qwen_model, current_prompt, split_str, temperature=0.0, stop_tokens=["#################", "###"], max_len=3)
            
                        if "yes" in res.lower():
                            flag_set.add(event_id_tmp)
                            event_id_group.append(event_id_tmp)
                            len_1, len_2 = len(content_1.split(" ")), len(content_2.split(" "))
                            if len_1 < 12 and len_2 < 12:
                                rep_content = content_1 if (len_1 > len_2) else content_2
                                rep_id = event_id if (len_1 > len_2) else event_id_tmp
                            elif len_1 < 12 and len_2 >= 12:
                                rep_content = content_2
                                rep_id = event_id_tmp
                            elif len_1 >= 12 and len_2 < 12:
                                rep_content = content_1
                                rep_id = event_id
                            else:
                                rep_content = content_2 if (len_1 > len_2) else content_1
                                rep_id = event_id_tmp if (len_1 > len_2) else event_id
                            relation_event_pairs = [content_1, content_2]
                            with open(os.path.join(dir_path, "event_relation_fine_grained.jsonl"), "a", encoding="utf-8") as f:
                                f.write(json.dumps(relation_event_pairs, ensure_ascii=False) + "\n")
                    if rep_id == event_id:
                        event_id_group_dic[rep_id] = event_id_group
                    else:
                        if event_id in event_id_group_dic.keys():
                            del event_id_group_dic[event_id]
                        event_id_group_dic[rep_id] = event_id_group
                    
            if event_id_list_len > 15:
                client = chromadb.Client()
                collection = client.create_collection(name=f"{keyword}-my_collection", metadata={"hnsw:space": "cosine"})
                event_id_tmp_list = list(event_id_group_dic.keys())
                content_list = []
                embeddings = []
                for event_id_tmp in event_id_group_dic.keys():
                    content_list.append(id2event[event_id_tmp])
                    embedding = gte_model.encode(id2event[event_id_tmp])
                    embeddings.append(embedding.tolist())
                
                list_len = len(event_id_tmp_list)
                i = 0
                while i < list_len:
                    if i + 5000 >= list_len:
                        collection.add(
                            embeddings=embeddings[i: list_len],
                            documents=content_list[i: list_len],
                            metadatas=[{"event_id": e_id} for e_id in event_id_tmp_list[i: list_len]],
                            ids=[str(i_) for i_ in list(range(i, list_len))]
                        )
                        break
                    else:
                        collection.add(
                            embeddings=embeddings[i: i + 5000],
                            documents=content_list[i: i + 5000],
                            metadatas=[{"event_id": e_id} for e_id in event_id_tmp_list[i: i + 5000]],
                            ids = [str(i_) for i_ in list(range(i, i + 5000))]
                        )
                        i += 5000
                    
                for e_idx_1 in tqdm(range(15, event_id_list_len), desc=f"{keyword}-long length"):
                    event_id = event_id_list[e_idx_1]
                    content_1 = id2event[event_id]
                    
                    embedding = gte_model.encode(content_1)
                    embeddings = [embedding.tolist()]
                    n_results = min(3, list_len)
                    results = collection.query(
                        query_embeddings=embeddings,  # 查询文本
                        n_results=n_results,               # 返回最相似的3个结果
                        include=["documents", "distances", "metadatas"]  # 返回的内容
                    )

                    similiar_events = results["documents"][0]
                    similiar_event_ids = [re["event_id"] for re in results["metadatas"][0]]

                    flag = 0
                    for event_id_tmp, event_content in zip(similiar_event_ids, similiar_events):
                        current_prompt = prompt.format(input_1=content_1, input_2=event_content)
                        res = chat_completions4(current_prompt, split_str)
                        #res = completion_with_llm(Qwen_tokenizer, Qwen_model, current_prompt, split_str, temperature=0.0, stop_tokens=["#################", "###"], max_len=3)
                        if "no" in res.lower():
                            continue
                        if "yes" in res.lower():
                            flag = 1
                            len_1, len_2 = len(content_1.split(" ")), len(event_content.split(" "))
                            if len_1 < 12 and len_2 < 12:
                                rep_content = content_1 if (len_1 > len_2) else event_content
                                rep_id = event_id if (len_1 > len_2) else event_id_tmp
                            elif len_1 < 12 and len_2 >= 12:
                                rep_content = event_content
                                rep_id = event_id_tmp
                            elif len_1 >= 12 and len_2 < 12:
                                rep_content = content_1
                                rep_id = event_id
                            else:
                                rep_content = event_content if (len_1 > len_2) else content_1
                                rep_id = event_id_tmp if (len_1 > len_2) else event_id

                            if rep_id == event_id_tmp:
                                if event_id_tmp in event_id_group_dic.keys():
                                    event_id_group_dic[event_id_tmp].append(event_id)
                                else:
                                    for rep_id_tmp, id_list in event_id_group_dic.items():
                                        if event_id_tmp in id_list:
                                            event_id_group_dic[rep_id_tmp].append(event_id)
                                            break
                            else:
                                if event_id_tmp in event_id_group_dic.keys():
                                    event_id_group_dic[rep_id] = event_id_group_dic[event_id_tmp]
                                else:
                                    for rep_id_tmp, id_list in event_id_group_dic.items():
                                        if event_id_tmp in id_list:
                                            event_id_group_dic[rep_id_tmp].append(event_id)
                                            rep_id = rep_id_tmp
                                            break
                                event_id_group_dic[rep_id].append(rep_id)
                                event_id_group_dic[rep_id] = list(set(event_id_group_dic[rep_id]))
                                if event_id_tmp in event_id_group_dic.keys():
                                    del event_id_group_dic[event_id_tmp]
                                    event_id_tmp_list = list(set(list(event_id_group_dic.keys())))
                                    content_list = []
                                    embeddings_tmp = []
                                    for e_id in event_id_tmp_list:
                                        content_list.append(id2event[e_id])
                                        embedding = gte_model.encode(id2event[e_id])
                                        embeddings_tmp.append(embedding.tolist())
                                    
                                    batch_size = 5000  # 选择一个安全的批量大小，比如5000
                                    for i in range(0, len(content_list), batch_size):
                                        collection.update(
                                            embeddings=embeddings_tmp[i:i+batch_size],
                                            documents=content_list[i:i+batch_size],
                                            metadatas=[{"event_id": e_id} for e_id in event_id_tmp_list[i:i+batch_size]],
                                            ids=[str(i_) for i_ in range(i, min(i+batch_size, len(content_list)))]
                                        )

                            relation_event_pairs = [content_1, event_content]
                            with open(os.path.join(dir_path, "event_relation_fine_grained.jsonl"), "a", encoding="utf-8") as f:
                                f.write(json.dumps(relation_event_pairs, ensure_ascii=False) + "\n")
                            
                            break

                    if flag == 0:
                        event_id_group_dic[event_id] = [event_id]
                        count = len(event_id_group_dic)
                        collection.add(
                            embeddings=embeddings,
                            documents=[content_1],
                            metadatas=[{"event_id": event_id}],
                            ids=[str(count)]
                        )

                client.delete_collection(name=f"{keyword}-my_collection")

            event_id_group_dic_tmp = copy.deepcopy(event_id_group_dic)
            for key, values in event_id_group_dic_tmp.items():
                event_id_group_dic[key] = list(set(values))

            cluster_splitting(cls_id, event_id_group_dic, event_pool, event2cluster)

    event_pool_tmp = copy.deepcopy(event_pool)
    for cls_id, event_id_list in event_pool_tmp.items():
        for event_id in event_id_list:
            if event2cluster[event_id] != cls_id:
                neibor_cls_id = event2cluster[event_id]
                for neibor_event_id in event_pool_tmp[neibor_cls_id]:
                    event2cluster[neibor_event_id] = cls_id
                event_pool[cls_id] = list(set(event_pool[cls_id] + event_pool[neibor_cls_id]))
                if neibor_cls_id in event_pool.keys():
                    del event_pool[neibor_cls_id]

    with open(os.path.join(dir_path, "event_pool_relation_fine_grained.json"), "w", encoding="utf-8") as f:
        json.dump(event_pool, f, ensure_ascii=False, indent=4)
    
    with open(os.path.join(dir_path, "event2cluster_relation_fine_grained.json"), "w", encoding="utf-8") as f:
        json.dump(event2cluster, f, ensure_ascii=False, indent=4)


def same_cluster_splitting(dir_path, dataset, keyword):
    if dataset == "t17":
        if k != "h1n1":
            prompt = RELATION_CLUSTER_SPLIT_PROMPT
        else:
            prompt = RELATION_CLUSTER_SPLIT_PROMPT_TMP
    
    split_str = "### The determination of whether the above two statements are the same event is"

    with open(os.path.join(dir_path, "event_pool_relation_fine_grained.json"), "r", encoding="utf-8") as f:
        event_pool = json.load(f)
    
    with open(os.path.join(dir_path, "event2cluster_relation_fine_grained.json"), "r", encoding="utf-8") as f:
        event2cluster = json.load(f)

    with open(os.path.join(dir_path, "id2event.json"), "r", encoding="utf-8") as f:
        id2event = json.load(f)
    
    event_pool = {int(key): value for key, value in event_pool.items()}
    event2cluster = {int(key): value for key, value in event2cluster.items()}
    id2event = {int(key): value for key, value in id2event.items()}

    event_pool_tmp = copy.deepcopy(event_pool)
    for cls_id, event_id_list in tqdm(event_pool_tmp.items(), desc=f"acquire same event clusters:{dataset}-{keyword}"):
        if len(event_id_list) > 1:
            event_id_group_dic = dict()
            flag_set = set()
            
            event_id_list_len = len(event_id_list)
            for e_idx_1 in range(15):
                if e_idx_1 < event_id_list_len:
                    event_id = event_id_list[e_idx_1]
                    content_1 = id2event[event_id]
                    if event_id in flag_set:
                        continue
                    flag_set.add(event_id)
                    event_id_group = [event_id]
                    event_id_group_dic[event_id] = event_id_group

                    for e_idx_2 in range(e_idx_1 + 1, 15):
                        if e_idx_2 >= event_id_list_len:
                            break
                        event_id_tmp = event_id_list[e_idx_2]
                        content_2 = id2event[event_id_tmp]
                        if event_id_tmp in flag_set:
                            continue
                        current_prompt = prompt.format(input_1=content_1, input_2=content_2)
                        res = chat_completions4(current_prompt, split_str)
                        # res = completion_with_llm(Qwen_tokenizer, Qwen_model, current_prompt, split_str, temperature=0.0, stop_tokens=["#################", "###"], max_len=3)
            
                        if "yes" in res.lower():
                            flag_set.add(event_id_tmp)
                            event_id_group.append(event_id_tmp)

                    event_id_group_dic[event_id] = event_id_group        

            if event_id_list_len > 15:
                for e_idx_1 in range(15, event_id_list_len):
                    event_id = event_id_list[e_idx_1]
                    content_1 = id2event[event_id]
                    
                    flag = 0
                    for event_neibor_id in event_id_group_dic.keys():
                        content_2 = id2event[event_neibor_id]
                        current_prompt = prompt.format(input_1=content_1, input_2=content_2)
                        res = chat_completions4(current_prompt, split_str)

                        if "yes" in res.lower():
                            flag = 1
                            event_id_group_dic[event_neibor_id].append(event_id)
                            break

                    if flag == 0:
                        event_id_group_dic[event_id] = [event_id]

            cluster_splitting(cls_id, event_id_group_dic, event_pool, event2cluster)   

    with open(os.path.join(dir_path, "event_pool_same_events.json"), "w", encoding="utf-8") as f:
        json.dump(event_pool, f, ensure_ascii=False, indent=4)
    
    with open(os.path.join(dir_path, "event2cluster_same_events.json"), "w", encoding="utf-8") as f:
        json.dump(event2cluster, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    args = get_argparser()

    if args.keyword == "all":
        keyword = [name for name in os.listdir(dataset_dir) 
                if os.path.isdir(os.path.join(dataset_dir, name))]
    else:
        keyword = args.keyword.split(',')

    device = torch.device("cuda:1")
    for k in keyword:
        dir_path = f"./processing/{args.dataset}/{k}/"

        if args.dataset == "t17":
            if k != "haiti":
                prompt = RELATION_STATEMENTS_SUMMARY_PROMPT
            else:
                prompt = RELATION_STATEMENTS_SUMMARY_PROMPT_TMP
        
        relation_events_summarization(dir_path, args.dataset, k, prompt)
        id2events_construction(dir_path, args.dataset, k)
        relation_events_clustering(dir_path, args.dataset, k)

        relation_cluster_splitting(dir_path, args.dataset, k, device)
        same_cluster_splitting(dir_path, args.dataset, k)