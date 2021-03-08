from config import Config
data_path = Config().processed_data
with open(data_path + 'new_train.txt', 'w', encoding='utf8') as fw:
    with open(data_path + 'train.txt', 'r', encoding='utf8') as fr:
        for line in fr:
            fw.write(line)

    with open(data_path + 'test.txt', 'r', encoding='utf8') as fr:
        for line in fr:
            fw.write(line)
