import json
import os
from tqdm import tqdm


def find_sublist_index(words_text, trigger_list):
    n = len(trigger_list)
    for i in range(len(words_text) - n + 1):
        if words_text[i:i+n] == trigger_list:
            return i
    return -1


def trigger_statistics(file_path, trigger_pos_dic):
    with open(file_path, "r", encoding='utf-8') as f:
        data_list = json.load(f)

    dataset_name = file_path.split("_")[0]
    for d in tqdm(data_list, desc=f"{dataset_name} pos statistics"):
        words_text = d["words_text"]
        words_text = " ".join(words_text).lower().split(" ")
        words_pos = d["words_pos"]
        trigger_info = d["trigger_info"]["triggers"]
        #trigger_info.extend(d["trigger_info"]["negative_triggers"])
        for t in trigger_info:
            trigger = t["trigger"]
            trigger_list = trigger.strip().split(" ")
            trigger_len = len(trigger_list)
            if trigger_len == 1 and trigger in words_text:
                index = words_text.index(trigger)
                w_pos = words_pos[index]
                if w_pos not in trigger_pos_dic.keys():
                    trigger_pos_dic[w_pos] = dict()
                    trigger_pos_dic[w_pos]["count"] = 0
                    trigger_pos_dic[w_pos]["trigger"] = trigger
                    trigger_pos_dic[w_pos]["sentence"] = d["sentence"]
                trigger_pos_dic[w_pos]["count"] += 1
            elif trigger_len > 1:
                head_index = find_sublist_index(words_text, trigger_list)
                if head_index == -1:
                    continue
                t_pos_list = words_pos[head_index: head_index + trigger_len]
                t_pos = " ".join(t_pos_list)
                if t_pos not in trigger_pos_dic.keys():
                    trigger_pos_dic[t_pos] = dict()
                    trigger_pos_dic[t_pos]["count"] = 0
                    trigger_pos_dic[t_pos]["trigger"] = trigger
                    trigger_pos_dic[t_pos]["sentence"] = d["sentence"]
                trigger_pos_dic[t_pos]["count"] += 1


if __name__ == "__main__":
    des_dir = r'./MAVEN'
    valid_file_path = r'./MAVEN/valid_trigger_post_process_pos.json'
    train_file_path = r'./MAVEN/train_trigger_post_process_pos.json'
    trigger_pos_dic = dict()
    trigger_statistics(valid_file_path, trigger_pos_dic)
    trigger_statistics(train_file_path, trigger_pos_dic)
    pos_statistics_file = r"./MAVEN/trigger_pos_statistics.json"
    with open(pos_statistics_file, "w", encoding='utf-8') as f:
        json.dump(trigger_pos_dic, f, ensure_ascii=False, indent=4)