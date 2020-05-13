#train dev test 前の章で作ったものを使う
#full_data_tokened.txtは3つの全部が入ってる
#mozes でトークナイズ済み
import os 

cu_path = os.getcwd()
data_path = os.path.join(cu_path,'data','full_data_tokened.txt')

vocab = {}
with open(data_path) as data:
    for line in data:
        text_data = line.split("\t")[0]
        words = text_data.split()

        for word in  words:
            if word in vocab:
                vocab[word] +=1
            else:
                vocab[word] = 1

#あとでサンプルで使う
sample = words

#(単語,頻度)の順にタプルで入ってる
vocab_sorted = sorted(vocab.items(),key=lambda x:x[1])[::-1]

vocab_with_id = {}
#idに変換
for item in vocab_sorted:
    if item[1] >= 2:
        vocab_with_id[item[0]] = len(vocab_with_id.items())
    else:
        vocab_with_id[item[0]] = 0

def word2id(words):
    return [vocab_with_id[word] for word in words]

#trainのみで作る
#devとかtestのみに出現する単語は0にしてねみたいな処理をする

#sample
print(vocab_with_id)
print(sample)
print(word2id(sample))






        

        
        
        