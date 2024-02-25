import json
import numpy as np

from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def eval_language(preds, gts, task):
    # 计算BLEU得分
    bleu_score = corpus_bleu(gts, preds)
    print(str(task), "BLEU score:", bleu_score)

    # 计算METEOR得分
    assert len(gts) == len(preds)
    num, m_score = len(gts), 0.0
    for i in range(num):
        me_gt = [gts[i][0].replace(',', '').replace('.', '').split(' ')]
        me_pred = preds[i].replace(',', '').replace('.', '').split(' ')
        m_score += meteor_score(me_gt, me_pred)
    print(str(task), "METEOR score:", m_score / num)

    preds_dict, gts_dict = {}, {}
    for i in range(len(gts)):
        preds_dict[int(i)] = [preds[i]]
        gts_dict[int(i)] = gts[i]

    # 计算ROUGE-N、ROUGE-L和ROUGE-W得分
    rouge_eval = Rouge()
    rouge_score, _ = rouge_eval.compute_score(gts_dict, preds_dict)
    print(str(task), "ROUGE score:", rouge_score)

    # 计算 CIDEr 分数
    cider_eval = Cider()
    cider_score, _ = cider_eval.compute_score(gts_dict, preds_dict)
    print(str(task), "CIDEr score:", cider_score)


def eval_relation(preds, gts):
    # 计算准确率
    accuracy = accuracy_score(gts, preds)
    print("Accuracy:", accuracy)

    # 计算精确率
    precision = precision_score(gts, preds, average='micro')
    print("Precision:", precision)

    # 计算召回率
    recall = recall_score(gts, preds, average='micro')
    print("Recall:", recall)

    # 计算F1值
    f1 = f1_score(gts, preds, average='micro')
    print("F1 Score:", f1)


def main():
    summary_path = 'output/pred_results/summary_results.json'
    caption_path = 'output/pred_results/caption_results.json'
    relation_path = 'output/pred_results/relation_results.json'
    with open(summary_path, 'r') as f:
        summary_data = json.load(f)
    with open(caption_path, 'r') as f:
        caption_data = json.load(f)
    with open(relation_path, 'r') as f:
        relation_data = json.load(f)
    
    summary_preds = summary_data['preds']
    summary_gts = summary_data['gts']
    summary_gts = [[gt] for gt in summary_gts]
    eval_language(summary_preds, summary_gts, task='summary')

    caption_preds = caption_data['preds']
    caption_gts = caption_data['gts']
    caption_gts = [[gt] for gt in caption_gts]
    eval_language(caption_preds, caption_gts, task='caption')

    relation_preds = relation_data['preds']
    filtererd_relation_preds = []
    for relation_pred in relation_preds:
        filtererd_relation_pred = [1 if x > 0.4 else 0 for x in relation_pred]
        filtererd_relation_preds.append(filtererd_relation_pred)
    eval_relation(filtererd_relation_preds, relation_data['gts'])

def check():
    gts = ['a girl dressed in white and black, dancing in the center.'.replace(',', '').replace('.', '').split(' ')]
    preds = 'a girl dressed in white and black, dancing in the center.'.replace(',', '').replace('.', '').split(' ')
    m_score = meteor_score(gts, preds)
    print("METEOR score:", m_score)


if __name__ == '__main__':
    # nltk.download('wordnet')
    main()
    # check()
