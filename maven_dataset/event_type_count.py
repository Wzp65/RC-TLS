import json
import os
from tqdm import tqdm

def post_precessing():
    des_dir = r'./MAVEN'
    train_file_path = os.path.join(des_dir, "train.jsonl")
    valid_file_path = os.path.join(des_dir, "valid.jsonl")
    train_data = []
    with open(train_file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="train triggers post processing"):
            data = json.loads(line.strip())
            content_list = data['content']
            sent_list = []
            
            train_dic_temp = dict()
            for content in content_list:
                sent_list.append(content['sentence'])
            for event in data["events"]:
                event_type = event['type']
                for mention in event["mention"]:
                    trigger_word = mention["trigger_word"]
                    sent_id = mention['sent_id']
                    sentence = sent_list[sent_id]
                    if sentence not in train_dic_temp.keys():
                        train_dic_temp[sentence] = dict()
                        train_dic_temp[sentence]['triggers'] = []
                        train_dic_temp[sentence]['negative_triggers'] = []
                    train_dic = dict()
                    train_dic['event_type'] = event_type
                    train_dic['trigger'] = trigger_word
                    train_dic_temp[sentence]['triggers'].append(train_dic)
                    
            for negative_info in data['negative_triggers']:
                #print(negative_info)
                event_type = "trigger_none"
                sent_id = negative_info['sent_id']
                sentence = sent_list[sent_id]
                if sentence not in train_dic_temp.keys():
                    train_dic_temp[sentence] = dict()
                    train_dic_temp[sentence]['triggers'] = []
                    train_dic_temp[sentence]['negative_triggers'] = []
                train_dic = dict()
                train_dic['event_type'] = event_type
                train_dic['trigger'] = negative_info['trigger_word']
                train_dic_temp[sentence]['negative_triggers'].append(train_dic)

            for sent, item in train_dic_temp.items():
                train_dic = dict()
                train_dic["sentence"] = sent
                train_dic["trigger_info"] = item
                train_data.append(train_dic)

    train_file_path_post = os.path.join(des_dir, "train_trigger_post_process.json")
    with open(train_file_path_post, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=True, indent=4)

    train_data = []
    with open(valid_file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="valid triggers post processing"):
            data = json.loads(line.strip())
            content_list = data['content']
            sent_list = []
            
            train_dic_temp = dict()
            for content in content_list:
                sent_list.append(content['sentence'])
            for event in data["events"]:
                event_type = event['type']
                for mention in event["mention"]:
                    trigger_word = mention["trigger_word"]
                    sent_id = mention['sent_id']
                    sentence = sent_list[sent_id]
                    if sentence not in train_dic_temp.keys():
                        train_dic_temp[sentence] = dict()
                        train_dic_temp[sentence]['triggers'] = []
                        train_dic_temp[sentence]['negative_triggers'] = []
                    train_dic = dict()
                    train_dic['event_type'] = event_type
                    train_dic['trigger'] = trigger_word
                    train_dic_temp[sentence]['triggers'].append(train_dic)
                    
            for negative_info in data['negative_triggers']:
                #print(negative_info)
                event_type = "trigger_none"
                sent_id = negative_info['sent_id']
                sentence = sent_list[sent_id]
                if sentence not in train_dic_temp.keys():
                    train_dic_temp[sentence] = dict()
                    train_dic_temp[sentence]['triggers'] = []
                    train_dic_temp[sentence]['negative_triggers'] = []
                train_dic = dict()
                train_dic['event_type'] = event_type
                train_dic['trigger'] = negative_info['trigger_word']
                train_dic_temp[sentence]['negative_triggers'].append(train_dic)

            for sent, item in train_dic_temp.items():
                train_dic = dict()
                train_dic["sentence"] = sent
                train_dic["trigger_info"] = item
                train_data.append(train_dic)

    valid_file_path_post = os.path.join(des_dir, "valid_trigger_post_process.json")
    with open(valid_file_path_post, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=True, indent=4)


def post_statistics():
    des_dir = r'./MAVEN'
    train_file_path = os.path.join(des_dir, "train_trigger_post_process.json")
    valid_file_path = os.path.join(des_dir, "valid_trigger_post_process.json")
    type_statistics = dict()
    with open(train_file_path, 'r', encoding='utf-8') as f:
        train_data_list = json.load(f)
    for train_data in train_data_list:
        triggers_info = train_data['trigger_info']['triggers']
        for trigger_info in triggers_info:
            event_type = trigger_info['event_type']
            trigger = trigger_info['trigger']
            if event_type not in type_statistics.keys():
                type_statistics[event_type] = 0
            type_statistics[event_type] += 1
    
    with open(valid_file_path, 'r', encoding='utf-8') as f:
        valid_data_list = json.load(f)
    for train_data in valid_data_list:
        triggers_info = train_data['trigger_info']['triggers']
        for trigger_info in triggers_info:
            event_type = trigger_info['event_type']
            trigger = trigger_info['trigger']
            if event_type not in type_statistics.keys():
                type_statistics[event_type] = 0
            type_statistics[event_type] += 1
    
    type_statistics = {k: v for k, v in sorted(type_statistics.items(), key=lambda item: item[1], reverse=True)}
    post_statistics_file = os.path.join(des_dir, "type_statistics.json")
    with open(post_statistics_file, 'w', encoding='utf-8') as f:
        json.dump(type_statistics, f, ensure_ascii=True, indent=4)


if __name__ == "__main__":
    post_precessing()
    post_statistics()