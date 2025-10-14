import os
import json
from tqdm import tqdm
import argparse
import copy

import torch

from datetime import datetime, date, timedelta
import arrow

from openai import OpenAI
import openai
import requests

from chromadb import PersistentClient
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from prompt_template import SAME_CLUSTER_SUMMARIZE_PROMPT_TMP, SAME_CLUSTER_SUMMARIZE_PROMPT, DAY_SUMMARIZE_PROMPT, DAY_SUMMARIZE_PROMPT_TMP
from utils import parse_time

from tilse.data.timelines import Timeline as TilseTimeline
from tilse.data.timelines import GroundTruth as TilseGroundTruth
from tilse.evaluation import rouge

from pprint import pprint
from evaluation import get_scores, evaluate_dates, get_average_results

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyword", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default="./datasets")
    parser.add_argument("--src_dir", type=str, default="./processing")
    parser.add_argument("--Qwen_model", type=str, default="/mnt/sdb1/wangzeping2023/Qwen/Qwen3-4B")
    parser.add_argument("--des_dir", type=str, default="./timeline")
    args = parser.parse_args()
    return args


def get_avg_score(scores):
    return sum(scores) / len(scores)


def save_json(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)


def date_to_summaries(timeline):
    timeline_dic = dict()
    for day_events_info in timeline:
        time = day_events_info[0]
        parse_gold_time = parse_time(time)
        date_time = parse_gold_time.date()
        timeline_dic[date_time] = day_events_info[1]

    return timeline_dic

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


def cluster_summarization(id2event, same_event_pool, same_event2cluster, dataset, keyword, des_dir):
    client = PersistentClient()
    collection = client.get_collection(name=f"gte_{dataset}_{keyword}")

    if args.dataset == "t17":
        if k != "h1n1":
            prompt = SAME_CLUSTER_SUMMARIZE_PROMPT_TMP
        else:
            prompt = SAME_CLUSTER_SUMMARIZE_PROMPT

    split_str = "### Based on the statements 2 above, write a concise summary of the main event part:"
    same_cls2event = dict()
    for cls_id, event_id_list in tqdm(same_event_pool.items(), desc=f"same event summarization:{dataset}-{keyword}"):
        if len(event_id_list) > 1:
            content_list = []
            pub_time = []
            event_time = []
            for event_id in event_id_list:
                content_list.append(id2event[event_id])
                results = collection.get(
                    where={"event_id": event_id}  # 元数据过滤条件
                )

                query_event_time = results["metadatas"][0]["time"]
                query_pubtime = results["metadatas"][0]["pubtime"]
                parse_query_event_time = parse_time(query_event_time)
                parse_query_pubtime = parse_time(query_pubtime)
                if parse_query_event_time == parse_query_pubtime:
                    pub_time.append(parse_query_pubtime)
                else:
                    event_time.append(parse_query_event_time)
            
            statements = "\n".join(content_list)
            current_prompt = prompt.format(statements=statements)
            res = chat_completions4(current_prompt, split_str)
            print(res)
            same_cls2event[cls_id] = dict()
            same_cls2event[cls_id]["content"] = res
            pub_time = sorted(pub_time)
            event_time = sorted(event_time)
            if len(event_time) > 0 and len(pub_time) > 0:
                if pub_time[0] > event_time[0] and (event_time[0].month != 1 and event_time[0].day != 1):
                    same_cls2event[cls_id]["time"] = event_time[0].strftime("%Y-%m-%dT%H:%M:%S")
                else:
                    same_cls2event[cls_id]["time"] = pub_time[0].strftime("%Y-%m-%dT%H:%M:%S")
            elif len(event_time) > 0 and len(pub_time) == 0:
                same_cls2event[cls_id]["time"] = event_time[0].strftime("%Y-%m-%dT%H:%M:%S")
            elif len(event_time) == 0 and len(pub_time) > 0:
                same_cls2event[cls_id]["time"] = pub_time[0].strftime("%Y-%m-%dT%H:%M:%S")

        else:
            same_cls2event[cls_id] = dict()
            same_cls2event[cls_id]["content"] = id2event[event_id_list[0]]
            results = collection.get(
                where={"event_id": event_id_list[0]}  # 元数据过滤条件
            )

            query_event_time = results["metadatas"][0]["time"]
            query_pubtime = results["metadatas"][0]["pubtime"]
            parse_query_event_time = parse_time(query_event_time)
            parse_query_pubtime = parse_time(query_pubtime)
            if parse_query_event_time == parse_query_pubtime:
                same_cls2event[cls_id]["time"] = query_pubtime
            else:
                if parse_query_pubtime > parse_query_event_time and (parse_query_event_time.month != 1 and parse_query_event_time.day != 1):
                    same_cls2event[cls_id]["time"] = query_event_time
                else:
                    same_cls2event[cls_id]["time"] = query_pubtime
    
    with open(os.path.join(des_dir, "same_cls2event.json"), "w", encoding="utf-8") as f:
        json.dump(same_cls2event, f, ensure_ascii=False, indent=4)


def acquire_sorted_event_info(id2event, same_event_pool, same_event2cluster, relation_event_pool, relation_event2cluster, dataset, keyword, des_dir, alpha):
    with open(os.path.join(des_dir, "same_cls2event.json"), "r", encoding="utf-8") as f:
        same_cls2event = json.load(f)
    
    same_cls2event = {int(key): value for key, value in same_cls2event.items()}

    for cls_id, event_id_list in same_event_pool.items():
        relation_cls_id = relation_event2cluster[event_id_list[0]]
        same_cls2event[cls_id]["relation_cls_id"] = relation_cls_id
        same_count = len(event_id_list)
        same_cls2event[cls_id]["same_event_count"] = same_count
        relation_count = len(relation_event_pool[relation_cls_id])
        same_cls2event[cls_id]["related_event_count"] = relation_count

    sorted_same_cls2event = sorted(same_cls2event.items(), key=lambda x: (alpha * x[1]["related_event_count"] + (1-alpha) * x[1]["same_event_count"]), reverse=True)
    sorted_same_cls2event = dict(sorted_same_cls2event)
    all_count = len(sorted_same_cls2event)

    with open(os.path.join(des_dir, "same_cls2event_info.json"), "w", encoding="utf-8") as f:
        json.dump(sorted_same_cls2event, f, ensure_ascii=False, indent=4)


def get_average_summary_length(ref_tl):
    lens = [len(summary[1]) for summary in ref_tl]
    return round(sum(lens) / len(lens))


def acquire_start_end_time(golden_timelines):
    g_start_time, g_end_time = [], []
    start_time, end_time = golden_timelines[0][0][0], golden_timelines[0][-1][0]
    
    for timeline in golden_timelines:
        g_start_time.append(timeline[0][0])
        g_end_time.append(timeline[-1][0])
        if timeline[0][0] < start_time:
            start_time = timeline[0][0]
        if timeline[-1][0] > end_time:
            end_time = timeline[-1][0] 
    
    return g_start_time, g_end_time, start_time, end_time


def filtered_timelines(golden_timelines, timeline_event_info, dataset, keyword, des_dir):
    g_start_time, g_end_time, start_time, end_time = acquire_start_end_time(golden_timelines)
    parse_start_time = parse_time(start_time)
    parse_end_time = parse_time(end_time)

    timeline_event_info_tmp = copy.deepcopy(timeline_event_info)
    for cls_id, event_info in timeline_event_info_tmp.items():
        try:
            parser_event_time = parse_time(event_info["time"])
        except ValueError:
            continue

        if parser_event_time < parse_start_time:
            del timeline_event_info[cls_id]
        if parser_event_time > parse_end_time:
            del timeline_event_info[cls_id]

    time_to_event = dict()
    for cls_id, event_info in timeline_event_info.items():
        event_info_time = event_info["time"]
        if event_info_time not in time_to_event.keys():
            time_to_event[event_info_time] = []
        time_to_event[event_info_time].append(event_info["content"])

    results = []
    pred_timelines = []
    for tl_index, (start_time, end_time) in enumerate(zip(g_start_time, g_end_time)):
        tilse_timeline = dict()
        parse_start_time = parse_time(start_time)
        parse_end_time = parse_time(end_time)
        golden_timeline = golden_timelines[tl_index]
        summary_length = get_average_summary_length(golden_timeline)
        print(keyword, str(summary_length))
        event_info_len = len(timeline_event_info)
        for index, (cls_id, event_info) in enumerate(timeline_event_info.items()):
            try:
                parser_event_time = parse_time(event_info["time"])
            except:
                continue
            '''
            if parser_event_time < parse_start_time and (parse_start_time - parser_event_time).days >= day_diff:
                continue
            if parser_event_time > parse_end_time and (parser_event_time - parse_end_time).days >= day_diff:
                continue
            '''
            if parser_event_time < parse_start_time:
                continue
            if parser_event_time > parse_end_time:
                continue
            event_time_date = parser_event_time.date()
            if event_time_date not in tilse_timeline.keys():
                tilse_timeline[event_time_date] = []
            if len(time_to_event[event_info["time"]]) == 1 and summary_length == 1 and index > int(event_info_len) - int(event_info_len / 8):
                continue
            if summary_length > 1:
                if summary_length > 3 and len(time_to_event[event_info["time"]]) <= 2:
                    if index > int(event_info_len / 3):
                        continue

                if len(time_to_event[event_info["time"]]) < summary_length:
                    if index > int(event_info_len / 2):
                        continue

                if len(time_to_event[event_info["time"]]) >= summary_length and len(time_to_event[event_info["time"]]) <= summary_length + 3:
                    if index > event_info_len - int(event_info_len / 6):
                        continue
                    
                    if len(tilse_timeline[event_time_date]) >= summary_length:
                        continue
                    
                if len(time_to_event[event_info["time"]]) > summary_length + 3:
                    if index > event_info_len - int(event_info_len / 6):
                        continue
                    if len(tilse_timeline[event_time_date]) >= summary_length + 1:
                        continue
                
            else:
                if len(time_to_event[event_info["time"]]) == 1:
                    if index > int(event_info_len / 3):
                        continue
                if len(time_to_event[event_info["time"]]) > summary_length and len(time_to_event[event_info["time"]]) <= summary_length + 2:
                    if index > event_info_len - int(event_info_len / 5):
                        continue
                    
                    if len(tilse_timeline[event_time_date]) >= summary_length:
                        continue
                    
                if len(time_to_event[event_info["time"]]) > summary_length + 2:
                    if index > event_info_len - int(event_info_len / 5):
                        continue
                    if len(tilse_timeline[event_time_date]) >= summary_length + 1:
                        continue
                

            tilse_timeline[event_time_date].append(event_info["content"])

        tilse_timeline_tmp = copy.deepcopy(tilse_timeline)
        for key, value in tilse_timeline_tmp.items():
            if len(value) == 0:
                del tilse_timeline[key]

        tilse_timeline_tmp = {key.strftime("%Y-%m-%d"): value for key, value in tilse_timeline.items()}

        pred_timelines.append(tilse_timeline_tmp)
    
    with open(os.path.join(des_dir, "pred_timlines.json"), "w", encoding="utf-8") as f:
        json.dump(pred_timelines, f, ensure_ascii=False, indent=4)
        

def evaluate_timelines(golden_timelines, dataset, keyword, des_dir):
    if os.path.exists(os.path.join(des_dir, "pred_timelines_info.json")):
        with open(os.path.join(des_dir, "pred_timelines_info.json"), "r", encoding="utf-8") as f:
            pred_timelines = json.load(f)
        pred_timelines_tmp = copy.deepcopy(pred_timelines)
        for pred_timeline_tmp, pred_timeline in zip(pred_timelines_tmp, pred_timelines):
            for event_time_str, events_list in pred_timeline_tmp.items():
                events_list = [event_c for event_c in events_list if event_c != ""]
                pred_timeline[event_time_str] = events_list
        
    else:
        with open(os.path.join(des_dir, "pred_timlines.json"), "r", encoding="utf-8") as f:
            pred_timelines = json.load(f)

    if dataset == "t17":
        if k == "haiti" or k == "iraq":
            prompt = DAY_SUMMARIZE_PROMPT
        else:
            prompt = DAY_SUMMARIZE_PROMPT_TMP

    assert len(pred_timelines) == len(golden_timelines)
    results = []
    pred_timelines_info = []
    split_str = "### Based on Statements 3, all simplified event statements placed on separate lines are:"
    for golden_timeline, pred_timeline in zip(tqdm(golden_timelines, desc=f"acquire timeline info:{dataset}-{keyword}"), pred_timelines):    

        tilse_timeline = {datetime.strptime(key, "%Y-%m-%d").date(): value for key, value in pred_timeline.items()}
        if not os.path.exists(os.path.join(des_dir, "pred_timelines_info.json")):
            tilse_timeline_tmp = copy.deepcopy(tilse_timeline)
            for event_time, event_content_list in tilse_timeline_tmp.items():
                day_events = "\n".join(event_content_list)
                current_prompt = prompt.format(statements=day_events)
                res = chat_completions4(current_prompt, split_str)
                print(res)
                res_list = res.split("\n")
                tilse_timeline[event_time] = res_list

        pred_timeline = TilseTimeline(tilse_timeline)
        ground_truth = TilseGroundTruth([TilseTimeline(date_to_summaries(golden_timeline))])    

        evaluator = rouge.TimelineRougeEvaluator(measures=["rouge_1", "rouge_2"])
        rouge_scores = get_scores(metric, pred_timeline, ground_truth, evaluator)
        date_scores = evaluate_dates(pred_timeline, ground_truth)
        timeline_res = (rouge_scores, date_scores, pred_timeline)
        results.append(timeline_res)
        tilse_timeline_tmp = {key.strftime("%Y-%m-%d"): value for key, value in tilse_timeline.items()}

        pred_timelines_info.append(tilse_timeline_tmp)
    
    trial_res = get_average_results(results)
    rouge_1 = trial_res[0]['f_score']
    trial_res[0]['f_score'] = round(rouge_1, 4)
    rouge_2 = trial_res[1]['f_score']
    trial_res[1]['f_score'] = round(rouge_2, 4)
    date_f1 = trial_res[2]['f_score']
    trial_res[2]['f_score'] = round(date_f1, 4)

    trial_save_path = des_dir
    save_json(trial_res, os.path.join(trial_save_path, 'avg_score.json'))

    with open(os.path.join(des_dir, "pred_timelines_info.json"), "w", encoding="utf-8") as f:
        json.dump(pred_timelines_info, f, ensure_ascii=False, indent=4)
    
    return rouge_1


if __name__ == "__main__":
    args = get_argparser()

    root_des_dir = os.path.join(args.des_dir, args.dataset)
    root_src_dir = os.path.join(args.src_dir, args.dataset)

    if args.keyword == "all":
        keyword = [name for name in os.listdir(root_src_dir) 
                if os.path.isdir(os.path.join(root_src_dir, name))]
    else:
        keyword = args.keyword.split(',')

    dataset_dir = os.path.join(args.dataset_path, args.dataset)

    r1_list = []
    r2_list = []
    date_list = []
    for k in tqdm(keyword, desc=f"acquire timeline and evaluate:{args.dataset}"):
        src_dir = os.path.join(root_src_dir, k)
        des_dir = os.path.join(root_des_dir, k)
        os.makedirs(des_dir, exist_ok=True)
        
        dataset_path = os.path.join(dataset_dir, k)

        id2event_path = os.path.join(src_dir, "id2event.json")
        with open(id2event_path, "r", encoding="utf-8") as f:
            id2event = json.load(f)
        same_event_pool_path = os.path.join(src_dir, "event_pool_same_events.json")
        with open(same_event_pool_path, "r", encoding="utf-8") as f:
            same_event_pool = json.load(f)
        same_event2cluster_path = os.path.join(src_dir, "event2cluster_same_events.json")
        with open(same_event2cluster_path, "r", encoding="utf-8") as f:
            same_event2cluster = json.load(f)
        relation_event_pool_path = os.path.join(src_dir, "event_pool_relation_fine_grained.json")
        with open(relation_event_pool_path, "r", encoding="utf-8") as f:
            relation_event_pool = json.load(f)
        relation_event2cluster_path = os.path.join(src_dir, "event2cluster_relation_fine_grained.json")
        with open(relation_event2cluster_path, "r", encoding="utf-8") as f:
            relation_event2cluster = json.load(f)
        
        same_event_pool = {int(key): value for key, value in same_event_pool.items()}
        same_event2cluster = {int(key): value for key, value in same_event2cluster.items()}
        relation_event_pool = {int(key): value for key, value in relation_event_pool.items()}
        relation_event2cluster = {int(key): value for key, value in relation_event2cluster.items()}
        id2event = {int(key): value for key, value in id2event.items()}

        cluster_summarization(id2event, same_event_pool, same_event2cluster, args.dataset, k, des_dir)
        
        acquire_sorted_event_info(id2event, same_event_pool, same_event2cluster, relation_event_pool, relation_event2cluster, args.dataset, k, des_dir, 0.5)
        
        with open(os.path.join(des_dir, "same_cls2event_info.json"), "r", encoding="utf-8") as f:
            event_info = json.load(f)
        count = len(event_info)

        timeline_count = int(0.08 * count)
        timeline_event_info = dict(list(event_info.items())[:timeline_count])

        sorted_timeline_event_info = sorted(timeline_event_info.items(), key=lambda x: x[1]["time"])
        sorted_timeline_event_info = dict(sorted_timeline_event_info)

        with open(os.path.join(des_dir, "timeline.json"), "w", encoding="utf-8") as f:
            json.dump(sorted_timeline_event_info, f, ensure_ascii=False, indent=4)
        
        
        

        evaluator = rouge.TimelineRougeEvaluator(measures=["rouge_1", "rouge_2"])
        metric = 'align_date_content_costs_many_to_one'
        
        golden_timelines_path = os.path.join(dataset_path, "timelines.jsonl")
        golden_timelines = []
        with open(golden_timelines_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                golden_timelines.append(data)

        
        filtered_timelines(golden_timelines, sorted_timeline_event_info, args.dataset, k, des_dir)

        evaluate_timelines(golden_timelines, args.dataset, k, des_dir)

        with open(os.path.join(des_dir, "avg_score.json"), "r", encoding="utf-8") as f:
            score_list = json.load(f)
        
        
        r1_list.append(score_list[0]["f_score"])
        r2_list.append(score_list[1]["f_score"])
        date_list.append(score_list[2]["f_score"])
    
    r1_f1 = get_avg_score(r1_list)
    r2_f1 = get_avg_score(r2_list)
    date_f1 = get_avg_score(date_list)

    results_dic = dict()
    results_dic["r1_f1"] = r1_f1
    results_dic["r2_f1"] = r2_f1
    results_dic["date_f1"] = date_f1

    with open(os.path.join(root_des_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results_dic, f, ensure_ascii=False, indent=4)
