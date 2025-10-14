import chromadb
import json
import os

# 连接到持久化Chroma客户端
client = chromadb.PersistentClient(path="./chroma")

# 获取所有集合名称
collection_names = [col.name for col in client.list_collections()]
print(collection_names)

file_1 = r'./processing/t17/mj/same_event.json'
file_2 = r'./processing/t17/mj/same_and_relation_events.json'
file_3 = r'./processing/t17/mj/id2event.json'
file_4 = r'./processing/t17/mj/same_events_summaries.json'
file_5 = r'./processing/t17/mj/same_events_summaries_1.json'

with open(file_1, "r") as f:
    list_1 = json.load(f)
with open(file_2, "r") as f:
    list_2 = json.load(f)

with open(file_3, "r") as f:
    list_3 = json.load(f)

with open(file_4, "r") as f:
    list_4 = json.load(f)

with open(file_5, "r") as f:
    list_5 = json.load(f)

print(len(list_1))
print(len(list_2))
print(len(list_3))
print(len(list_4))
print(len(list_5))