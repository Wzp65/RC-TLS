import os
import gzip
import json
import argparse
from tqdm import tqdm


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyword", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()
    return args


def read_jsonl_gz(file_path):
    article_data = []
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for line in f:
            article_data.append(json.loads(line))
    return article_data


def preprosess_data(dataset, keyword):
    dataset_dir = f'./datasets/{dataset}'
    if keyword == "all":
        keyword = [name for name in os.listdir(dataset_dir) 
                if os.path.isdir(os.path.join(dataset_dir, name))]
    else:
        keyword = keyword.split(',')
    
    for k in tqdm(keyword, desc=f"preprocess articles"):
        des_dir = os.path.join(dataset_dir, k)
        file_path = os.path.join(des_dir, "articles.preprocessed.jsonl.gz")
        article_data = read_jsonl_gz(file_path)
        keyword_sent_list = []
        for d in tqdm(article_data, desc=f"{k}"):
            title = d['title']
            pubtime = d['time']
            article_id = d['id']
            sent_list = d['sentences']
            sent_index = 0
            for sent_info in sent_list:
                sent = sent_info['raw']
                sent_time = sent_info['time'] if sent_info['time'] is not None else pubtime
                sent_dic = dict()
                sent_dic['title'] = title
                sent_dic['pubtime'] = pubtime
                sent_dic['article_id'] = article_id
                sent_dic['sent_index'] = sent_index
                sent_dic['content'] = sent
                sent_dic['time'] = sent_time
                keyword_sent_list.append(sent_dic)
                sent_index += 1

        preprocess_file_path = os.path.join(des_dir, "preprocess_to_sent.json")
        with open(preprocess_file_path, "w", encoding='utf-8') as file:
            json.dump(keyword_sent_list, file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    args = get_argparser()
    preprosess_data(args.dataset, args.keyword)
