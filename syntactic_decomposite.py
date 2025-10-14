import stanza
from stanza import DownloadMethod
import os
import json
from tqdm import tqdm
import argparse


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyword", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()
    return args


def decomposite_sents(data_sents, nlp, des_dir, keyword):
    batch_sents_info = []
    batch_sents = []
    data_list = []
    for d in tqdm(data_sents, desc=f"{keyword}"):
        batch_sents_info.append(d)
        batch_sents.append(d['content'])
        
        if len(batch_sents) == 256 or d == data_sents[-1]:
            docs = [nlp(sent) for sent in batch_sents]
            assert len(docs) == len(batch_sents_info)
            for doc, sent_info in zip(docs, batch_sents_info):
                sent = doc.sentences[0]
                words_text = []
                words_pos = []
                for word in sent.words:
                    words_text.append(word.text)
                    words_pos.append(word.pos)
                sent_info["words_text"] = words_text
                sent_info["words_pos"] = words_pos
                data_list.append(sent_info)
               
            batch_sents.clear()
            batch_sents_info.clear()

    pos_file_path = os.path.join(des_dir, "preprocess_to_sent.json")
    with open(pos_file_path, "w", encoding='utf-8') as f:
        json.dump(data_list, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    args = get_argparser()
    dataset_dir = f'./datasets/{args.dataset}'

    if args.keyword == "all":
        keyword = [name for name in os.listdir(dataset_dir) 
                if os.path.isdir(os.path.join(dataset_dir, name))]
    else:
        keyword = args.keyword.split(',')
    
    corenlp_dir = r"/mnt/sdb1/wangzeping2023/stanford-corenlp-4.5.10/"
    nlp = stanza.Pipeline(
        'en',
        processors='tokenize,pos,lemma,depparse',
        ssplit=False,  # 禁用自动分句（避免 Stanza 把多个句子合并）
        tokenize_batch_size=2056,
        pos_batch_size=2056,
        depparse_batch_size=2056,
        download_method=DownloadMethod.REUSE_RESOURCES
    )
    
    for k in tqdm(keyword, desc=f"decomposite process"):
        des_dir = os.path.join(dataset_dir, k)
        file_path = os.path.join(des_dir, "preprocess_to_sent.json")
        with open(file_path, "r", encoding='utf-8') as f:
            preprocess_data = json.load(f)
        decomposite_sents(preprocess_data, nlp, des_dir, k)
    