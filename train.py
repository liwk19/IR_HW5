from tqdm.auto import tqdm
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
import transformers
from transformers import BertTokenizer, BertModel, get_scheduler
import warnings
from torch import nn
from utils import *
import torch.nn.functional as F
import argparse
import copy


argparser = argparse.ArgumentParser("BERT IR", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
argparser.add_argument('--lr', type=float, default=1e-5)
argparser.add_argument('--num_epochs', type=int, default=3)
argparser.add_argument('--batch_size', type=int, default=60)
argparser.add_argument('--margin', type=float, default=1.0, help='used for triplet or triplet_new loss')
argparser.add_argument('--t', type=float, default=0.03, help='used for contrastive loss')
argparser.add_argument('--k1', type=float, default=0.2, help='used for combine loss')
argparser.add_argument('--k2', type=float, default=0.7, help='used for combine loss')
argparser.add_argument('--model_name', type=str, default='regression', 
    choices=['regression', 'contrastive', 'triplet', 'bm25_rerank', 'combine', 'triplet_new',
    'regression_rerank', 'contrastive_rerank', 'triplet_rerank'])
args = argparser.parse_args()


# 忽略不必要的输出
# warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()


def get_train_neg():
    samples = load_json('data/train_neg.json')
    return [[s['query'], s['positive_doc'], s['negative_doc']] for s in samples]


def get_train_score():
    samples = load_json('data/train_score.json')
    return [[s['query'], s['doc'], s['score']] for s in samples]


class TextEncoder(nn.Module):
    # 将文本编码为语义向量的编码器
    def __init__(self, path, device):
        # path初始为‘bert-base-chinese’，这是加载仓库里的模型，后面就变成了自己的存储路径
        super().__init__()
        self.tokenizer: BertTokenizer = BertTokenizer.from_pretrained(path)
        self.bert_model = BertModel.from_pretrained(path)
        self.device = device
        self.to(device)

    def forward(self, inputs_tuple):
        # inputs_tuple是一个元组，元组每个元素是一个形状为 [batch_size, seq_length] 的矩阵
        # 例如，对基于负样本的方法，有三个元素，分别表示：一个batch中的所有query, 一个batch中的所有对应的正例, 一个batch中的所有对应的难负例,
        # 模型把每个元素编码为 [batch_size, emb_dim] 的矩阵，emb_dim指语义向量的维度
        # 对基于算分数的方法，就两个元素，一个是query，一个是answer
        return [self.bert_model(**inputs)[0][:, 0] for inputs in inputs_tuple]

    def collate_fn_batch_neg(self, batch):
        queries, doc_pos, doc_neg = [x[0] for x in batch], [x[1] for x in batch], [x[2] for x in batch]
        queries_input = self.tokenizer(queries, padding=True, truncation=True, return_tensors="pt")
        pos_input = self.tokenizer(doc_pos, padding=True, truncation=True, return_tensors="pt")
        neg_input = self.tokenizer(doc_neg, padding=True, truncation=True, return_tensors="pt")
        return [queries_input, pos_input, neg_input]

    def collate_fn_pair_score(self, batch):
        queries, docs, scores = [x[0] for x in batch], [x[1] for x in batch], [x[2] for x in batch]
        queries_input = self.tokenizer(queries, padding=True, truncation=True, return_tensors="pt")
        docs_input = self.tokenizer(docs, padding=True, truncation=True, return_tensors="pt")
        scores = torch.tensor(scores)
        return [queries_input, docs_input, scores]

    def encode(self, texts):
        # TODO: 把原始文本列表texts编码成语义向量
        # texts是一个列表，列表里每个元素是一个字符串，需要编码
        interval = 1000
        text_embs = []
        with torch.no_grad():
            for i in range(0, len(texts), interval):
                high_bound = min(i+interval, len(texts))
                tokens = self.tokenizer(texts[i:high_bound], padding=True, truncation=True, return_tensors="pt")
                for k in tokens.keys():
                    tokens[k] = tokens[k].to(device)
                text_embs.append(self.bert_model(**tokens)[0][:, 0])
        text_embs = torch.cat(text_embs)
        assert text_embs.shape[0] == len(texts)
        return text_embs

    def save(self, path):
        self.tokenizer.save_pretrained(path)
        self.bert_model.save_pretrained(path)


class TripletLoss(nn.Module):
    def forward(self, outputs):
        # TODO: 实现的Triplet Loss训练损失
        # outputs里有三个矩阵A, B, C。(A[i], B[i], C[i])代表三元组(query, positive_doc, negative_doc)。
        # 参见 https://arxiv.org/pdf/1908.10084.pdf 的 Triplet Objective Function
        distance_pos = F.pairwise_distance(outputs[0], outputs[1], p=2)
        distance_neg = F.pairwise_distance(outputs[0], outputs[2], p=2)
        losses = F.relu(distance_pos - distance_neg + args.margin)
        return losses.mean()


class TripletNewLoss(nn.Module):
    def forward(self, outputs):
        # TODO: 实现的Triplet Loss训练损失
        # outputs里有三个矩阵A, B, C。(A[i], B[i], C[i])代表三元组(query, positive_doc, negative_doc)。
        # 参见 https://arxiv.org/pdf/1908.10084.pdf 的 Triplet Objective Function
        sim_pos = torch.cosine_similarity(outputs[0], outputs[1])
        sim_neg = torch.cosine_similarity(outputs[0], outputs[2])
        losses = F.relu(sim_neg - sim_pos + args.margin)
        return losses.mean()


class RegressionLoss(nn.Module):
    def forward(self, outputs, labels):
        # TODO: 实现相似度拟合的训练损失
        # 具体来说，outputs里有两个矩阵A, B。A，B的shape为[batch_size, 768]
        # (A[i], B[i], labels[i])代表三元组(query, doc, score)
        # 使A[i]和B[i]的相似度值与score接近，可以使用余弦相似度来衡量向量的相似度
        loss = nn.MSELoss()
        sim = torch.cosine_similarity(outputs[0], outputs[1])
        return loss(sim, labels)


class ContrastiveLoss(nn.Module):
    def forward(self, outputs):
        # TODO: 实现对比学习损失，考虑in-batch negatives和hard negative
        # 具体来说，outputs里有三个矩阵A, B, C。(A[i], B[i], C[i])代表三元组(query, positive_doc, negative_doc)。
        # 对于一个给定的A[k]：
        #   使A[k]与B[k]相似，而与B中其他向量不相似，即B中其他向量作为in-batch negatives；
        #   使A[k]与C[k]也不相似，即C[k]作为hard negative。
        # 可以使用余弦相似度来衡量向量的相似度。
        # 在实现上，A[k]与B、C矩阵各行的相似度最大值应在B[k]处产生，计算相似度后选择合适的损失函数来计算这一损失即可。
        query_norm = F.normalize(outputs[0], p=2, dim=1)
        answer = torch.cat([outputs[1], outputs[2]])
        answer_norm = F.normalize(answer, p=2, dim=1)
        scores = torch.mm(query_norm, answer_norm.T)
        labels = torch.tensor(range(len(scores)), dtype=torch.long, device=device)
        scores = scores / args.t
        loss = nn.CrossEntropyLoss()
        return loss(scores, labels)


# TODO: 实现用于rerank的Bert模型和训练
class TextPairScorer(nn.Module):
    def __init__(self, path, device):
        super().__init__()
        self.tokenizer: BertTokenizer = BertTokenizer.from_pretrained(path)
        self.bert_model = BertModel.from_pretrained(path)
        self.fc = nn.Linear(768, 1)
        self.device = device
        self.to(device)
    
    def forward(self, inputs):
        bert_emb = self.bert_model(**inputs[0])[0][:, 0]
        return self.fc(bert_emb).squeeze(1)

    def collate_fn_pair_score(self, batch):
        queries, docs, scores = [x[0] for x in batch], [x[1] for x in batch], [x[2] for x in batch]
        pair_input = self.tokenizer(queries, docs, padding=True, truncation=True, return_tensors="pt")
        scores = torch.tensor(scores)
        return [pair_input, scores]

    def encode(self, texts_pairs):
        interval = 1000
        text_embs = []
        with torch.no_grad():
            for i in range(0, len(texts_pairs), interval):
                high_bound = min(i+interval, len(texts_pairs))
                queries = [text[0] for text in texts_pairs[i:high_bound]]
                cands = [text[1] for text in texts_pairs[i:high_bound]]
                tokens = self.tokenizer(queries, cands, padding=True, truncation=True, return_tensors="pt")
                for k in tokens.keys():
                    tokens[k] = tokens[k].to(device)
                text_embs.append(self.forward([tokens]))
        text_embs = torch.cat(text_embs)
        assert text_embs.shape[0] == len(texts_pairs)
        return text_embs


def train(model, model_name=args.model_name):
    if model_name in ['regression', 'bm25_rerank', 'neural_rerank']:
        dataset = get_train_score()
        train_dataloader = DataLoader(dataset, batch_size=args.batch_size)
        train_dataloader.collate_fn = model.collate_fn_pair_score   # 就是做tokenize
        loss_fn = RegressionLoss() if model_name == 'regression' else nn.BCEWithLogitsLoss()
    elif model_name in ['contrastive', 'triplet', 'triplet_new']:
        dataset = get_train_neg()
        train_dataloader = DataLoader(dataset, batch_size=args.batch_size)
        train_dataloader.collate_fn = model.collate_fn_batch_neg
        if model_name == 'contrastive':
            loss_fn = ContrastiveLoss()
        elif model_name == 'triplet':
            loss_fn = TripletLoss()
        else:
            loss_fn = TripletNewLoss()
    elif model_name == 'combine':
        dataset1 = get_train_score()
        train_dataloader1 = DataLoader(dataset1, batch_size=args.batch_size)
        train_dataloader1.collate_fn = model.collate_fn_pair_score   # 就是做tokenize
        dataset2 = get_train_neg()
        train_dataloader2 = DataLoader(dataset2, batch_size=args.batch_size)
        train_dataloader2.collate_fn = model.collate_fn_batch_neg
        loss_reg = RegressionLoss()
        loss_con = ContrastiveLoss()
        loss_tri = TripletLoss()
    else:
        exit()

    # 定义训练策略
    if model_name == 'neural_rerank':   # 命令行只能指定1个lr，为ContrastiveLoss的lr，因此rerank的lr写死为最优值
        optimizer = AdamW(model.parameters(), lr=1e-5)
    else:
        optimizer = AdamW(model.parameters(), lr=args.lr)
    
    # 命令行只能指定1个num_epochs，为ContrastiveLoss的num_epochs，因此rerank的num_epochs写死为最优值
    if model_name == 'neural_rerank':
        args.num_epochs = 5
    if model_name != 'combine':
        num_training_steps = args.num_epochs * len(train_dataloader)
    else:
        num_training_steps = args.num_epochs * (len(train_dataloader1) + len(train_dataloader2))
    
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    print('training:', model_name)
    progress_bar = tqdm(range(num_training_steps))
    model.train()

    for epoch in range(args.num_epochs):
        if model_name != 'combine':
            for batch in train_dataloader:
                if model_name in ['regression', 'bm25_rerank', 'neural_rerank']:
                    batch, label = [{k: v.to(model.device) for k, v in input.items()} for input in batch[:-1]], batch[-1].to(model.device)
                    outputs = model(batch)
                    loss = loss_fn(outputs, label)
                elif model_name in ['contrastive', 'triplet', 'triplet_new']:
                    batch = [{k: v.to(model.device) for k, v in input.items()} for input in batch]
                    outputs = model(batch)
                    loss = loss_fn(outputs)
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
        
        elif model_name == 'combine':
            for batch in train_dataloader1:
                batch, label = [{k: v.to(model.device) for k, v in input.items()} for input in batch[:-1]], batch[-1].to(model.device)
                outputs = model(batch)
                loss = args.k1 * loss_reg(outputs, label)
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
            
            for batch in train_dataloader2:
                batch = [{k: v.to(model.device) for k, v in input.items()} for input in batch]
                outputs = model(batch)
                loss = args.k2 * loss_con(outputs) + (1-args.k1-args.k2) * loss_tri(outputs)
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)


def test(model, model_name=args.model_name, model2=None):
    quotes_ori = load_json('data/corpus.json')
    quotes = [quote['content'] for quote in quotes_ori]
    test_data = load_json('data/test_hard.json')
    model.eval()
    print('testing:', model_name)

    if model_name == 'bm25_rerank':   # 基于BM25进行重排序
        bm25_results_list = np.load('data/bm25_rank_list.npy', allow_pickle=True)
        rank_list = []
        for i in tqdm(range(len(test_data))):
            query = test_data[i]['query']
            answer_idx = quotes.index(test_data[i]['golden_quote'])
            bm25_results = np.array(bm25_results_list[i][0:100])
            texts_pairs = [[query, quotes[j]] for j in bm25_results]
            with torch.no_grad():
                scores = model.encode(texts_pairs)
            scores_rank = torch.argsort(scores, descending=True)
            scores_rank = np.array(scores_rank.cpu())
            scores_rank = bm25_results[scores_rank]
            goal = (scores_rank==answer_idx).nonzero()
            if goal[0].shape[0] == 1:
                rank_list.append(goal[0][0])
            elif goal[0].shape[0] == 0:
                rank_list.append((len(bm25_results) + 13200) / 2)
            else:
                exit()
        
        rank_list = np.array(rank_list) + 1
        recall_3 = (rank_list <= 3).mean()
        recall_10 = (rank_list <= 10).mean()
        recall_50 = (rank_list <= 50).mean()
        mrr = (1 / rank_list).mean()
    
    elif model_name == 'neural_rerank':   # 基于ContrastiveLoss进行重排序
        # 得到ContrastiveLoss的结果
        model.eval()
        with torch.no_grad():
            quotes_embeddings = model.encode(quotes)
        quotes_embeddings = F.normalize(quotes_embeddings, p=2, dim=-1)
        test_query = []
        for i in range(len(test_data)):
            test_query.append(test_data[i]['query'])
        with torch.no_grad():
            test_query = model.encode(test_query)
        test_query = F.normalize(test_query, p=2, dim=-1)
        scores = torch.mm(test_query, quotes_embeddings.T)   # shape: [207, 13201]
        scores_rank_ori = torch.argsort(scores, dim=1, descending=True)

        # 截取前100名结果进行重排序
        rank_list = []
        for i in tqdm(range(len(test_data))):
            query = test_data[i]['query']
            answer_idx = quotes.index(test_data[i]['golden_quote'])
            pre_results = scores_rank_ori[i][0:100]
            texts_pairs = [[query, quotes[j]] for j in pre_results]
            with torch.no_grad():
                scores = model2.encode(texts_pairs)
            scores_rank = torch.argsort(scores, descending=True)
            scores_rank = np.array(scores_rank.cpu())
            scores_rank = pre_results[scores_rank]
            goal = (scores_rank==answer_idx).nonzero()
            if goal.shape[0] != 0:
                rank_list.append(goal[0][0].item())
            else:
                rank_list.append((100 + 13200) / 2)
    
        rank_list = np.array(rank_list) + 1
        recall_3 = (rank_list <= 3).mean()
        recall_10 = (rank_list <= 10).mean()
        recall_50 = (rank_list <= 50).mean()
        mrr = (1 / rank_list).mean()

    else:
        # 将库中所有文本编码为向量
        model.eval()
        with torch.no_grad():
            quotes_embeddings = model.encode(quotes)
        quotes_embeddings = F.normalize(quotes_embeddings, p=2, dim=-1)
        test_query = []
        test_answer = []
        for i in range(len(test_data)):
            test_query.append(test_data[i]['query'])
            test_answer.append(quotes.index(test_data[i]['golden_quote']))
        with torch.no_grad():
            test_query = model.encode(test_query)
        test_query = F.normalize(test_query, p=2, dim=-1)
        scores = torch.mm(test_query, quotes_embeddings.T)   # shape: [207, 13201]
        
        scores_rank = torch.argsort(scores, dim=1, descending=True)
        scores_rank = scores_rank - torch.tensor(test_answer).unsqueeze(1).to(device)
        scores_goal = torch.argwhere(scores_rank == 0)[:, 1]
        recall_3 = (scores_goal < 3).sum() / scores_goal.shape[0]
        recall_10 = (scores_goal < 10).sum() / scores_goal.shape[0]
        recall_50 = (scores_goal < 50).sum() / scores_goal.shape[0]
        mrr = (1 / (scores_goal + 1)).mean()
    
    print(f'recall@3: {recall_3:.3f}, recall@10: {recall_10:.3f}, recall@50: {recall_50:.3f}, MRR: {mrr:.3f}')


def demo(model, query):
    # 将库中所有文本编码为向量
    quotes = [quote['content'] for quote in load_json("data/corpus.json")]
    quotes_embeddings = model.encode(quotes)
    quotes_embeddings = torch.nn.functional.normalize(quotes_embeddings, p=2, dim=-1)
    torch.save(quotes_embeddings, path+'/corpus_embedding')

    queries = ['要团结协作']
    for query in queries:
        # 将查询文本编码为向量
        query_embedding = model.encode([query])
        query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=-1)
        # 从库中找到与之最相似的向量
        scores = torch.mm(query_embedding, quotes_embeddings.t())[0]
        top_results = torch.topk(scores, k=min(5, len(quotes)))
        # 输出结果
        print(f'{query}:')
        for quote in [quotes[idx] for idx in top_results[1]]:
            print(f'  {quote}')


if __name__ == '__main__':
    set_seed(42)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if args.model_name == 'bm25_rerank':   # 基于BM25的结果做重排序
        model = TextPairScorer('bert-base-chinese', device)
        train(model)
        test(model)
    elif args.model_name in ['regression_rerank', 'contrastive_rerank', 'triplet_rerank']: # 基于神经网络结果做重排序
        model1 = TextEncoder('bert-base-chinese', device)
        train(model1, model_name=args.model_name.split('_')[0])
        model2 = TextPairScorer('bert-base-chinese', device)
        train(model2, model_name='neural_rerank')
        test(model1, model_name='neural_rerank', model2=model2)
    else:   # 3种损失函数，TripletNew损失函数，以及combine损失函数
        model = TextEncoder('bert-base-chinese', device)
        train(model)
        test(model)

    # demo(model, '时间是无价之宝')
