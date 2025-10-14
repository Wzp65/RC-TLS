# RC-TLS
Repo for paper "Retrieval-Augmented Event Detection Combined with A Novel Co-Referential Event Clustering Method: Qwen3-4B Powered New Paradigm for Timeline Summarization". 


## Timeine Summarization 

### Download T17 Dataset
To download datasets T17, please refer to [complementizer/news-tls ](https://github.com/complementizer/news-tls).

### Download MAVEN Dataset
To download MAVEN datasets, please refer to [THU-KEG/MAVEN-dataset ](https://github.com/THU-KEG/MAVEN-dataset).

### Workflow
1. To preprocess dataset articles to sentences, please refer to ```data_preprocess.py```
```
python data_preprocess.py --dataset "t17" --keyword "all"
```

2. To preprocess MAVEN dataset, please execute command ```cd ./maven_dataset``` and execute the following instruction steps in sequence:
```
python event_type_count.py
python syntactic_decomposite_maven.py --datatype "train"
python syntactic_decomposite_maven.py --datatype "valid"
python syntactic_statistics.py
python event_type_class.py
cd ../
```

3. Perform MAVEN chroma construction by ```create_chroma_bert_large.py```.
```
python create_chroma_bert_large.py
```

4. Preprocess T17 dataset to transform articles to sentences by ```data_preprocess.py```.
```
python data_preprocess.py --dataset "t17" --keyword "all"
```

5. Parse T17 sentences to POS taggers by ```syntactic_decomposite.py```.
```
python syntactic_decomposite.py --dataset "t17" --keyword "all"
```

6. Perform event detection by ```event_detection.py```.
```
python event_detection.py --dataset "t17" --keyword "all"
```

7. Perform clustering process by ```event_cluster.py```
```
python generate_clusters.py \
    --dataset "t17" \
    --keyword "all"
```

8. Perform timeline summarization and evaluation by ```timeline_summarization.py```
```
python timeline_summarization.py \
    --dataset "t17" \
    --keyword "all"
```
