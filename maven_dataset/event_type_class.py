import json
import os


def classify_event_type():
    train_file_path = r'./MAVEN/train_trigger_post_process.json'
    dev_file_path = r'./MAVEN/valid_trigger_post_process.json'

    with open(train_file_path, "r", encoding='utf-8') as f:
        train_data = json.load(f)
    
    with open(dev_file_path, "r", encoding='utf-8') as f:
        dev_data = json.load(f)
    
    event_type_dic = dict()
    for d in train_data:
        sentence = d["sentence"]
        trigers_info = d['trigger_info']['triggers']
        for t in trigers_info:
            event_type = t['event_type']
            trigger = t['trigger']
            if event_type not in event_type_dic.keys():
                event_type_dic[event_type] = []
            event_type_dic[event_type].append((sentence, trigger))

    for d in dev_data:
        sentence = d["sentence"]
        trigers_info = d['trigger_info']['triggers']
        for t in trigers_info:
            event_type = t['event_type']
            trigger = t['trigger']
            if event_type not in event_type_dic.keys():
                event_type_dic[event_type] = []
            event_type_dic[event_type].append((sentence, trigger))
    
    event_type_dic = {k: v for k, v in sorted(event_type_dic.items(), key=lambda item: len(item[1]), reverse=True)}
    store_file_path = r'./MAVEN/event_type_demon.json'
    with open(store_file_path, "w", encoding='utf-8') as f:
        json.dump(event_type_dic, f, ensure_ascii=False, indent=4)

classify_event_type()