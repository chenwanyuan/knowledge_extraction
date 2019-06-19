# knowledge_extraction  
结构化信息抽取，知识构建。  
源于第四届智能和语言大赛。  
最终结果是第18名。  
工作中的状态，只能利用业余时间，整个比赛就提交了3次结果，真希望下次有时间，好好搞一搞。
这是我第一次提交的baseline的方法。当时第一次提交是f1=82%, 第三次结果是86%

# 简介  
## 1提供一个知识抽取的代码。  
分为两个过程：1对schema进行分类，2针对schema进行信息抽取.  
## 2.两个过程都使用了bert。  
  1.schema进行分类。  
  使用了bert+ful_con+sigmoid （实际中我的实验还使用了另一种方法，下次再补充,）  
  2.针对schema进行信息抽取.  
  我把这个信息抽取的过程当成了阅读理解的多答案问题。  
  把schema的类型当成了问题（question），需要抽取的文章或句子当成了内容（paragraph），抽取的信息当成了答案(answer).  
  使用了bert+crf(相当于命名实体识别的方式进行抽取)  
## 3.提供服务。  
  针对训练的结果进行打包成graph_pb。  
  使用tornado框架提供抽取服务。  
# 备注  
信息抽取(Information Extraction, IE)是从自然语言文本中抽取实体、属性、关系及事件等事实类信息的文本处理技术，是信息检索、智能问答、智能对话等人工智
能应用的重要基础，一直受到业界的广泛关注。信息抽取任务涉及命名实体识别、指代消解、关系分类等复杂技术，极具挑战性。
本次竞赛发布基于schema约束的SPO信息抽取任务，即在给定schema集合下，从自然语言文本中抽取出符合schema要求的SPO三元组知识  
# 依赖  
Python==3.5  
TensorFlow==1.4.0  
pip install tornado  
pip install tqdm  
# 步骤  
## 下载bert模型  
下载bert的中文模型放在data/chinese_L-12_H-768_A-12  链接 [https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip]  
tar -xzvf corpus.tar.gz  
## schema分类器训练  
### 1.制作训练语料  
python classifier.py --type corpus  
### 2.训练模型  
python classifier.py --type train  
### 3.打包模型  
python classifier.py --type freeze  
## 知识抽取模型训练  
### 1.制作训练语料  
python extractor.py --type corpus  
### 2.训练模型  
python extractor.py --type train  
### 3.打包模型  
python extractor.py --type freeze  
# 启动  
## 启动服务  
bash script/start.sh  
## 停止服务  
bash script/stop.sh  
# 测试  
python test/client.py  
