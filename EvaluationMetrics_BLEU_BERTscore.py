import pandas as pd 
import numpy as np
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score

#all_reports = pd.read_csv("/Users/jessikabaral/Documents/MLHC/generated_reports.csv")
#all_reports = pd.read_csv("/Users/jessikabaral/Downloads/cxr-llava_custom_generated_reports_2.csv")
all_reports = pd.read_csv("/Users/jessikabaral/Downloads/cheXagent_custom_generated_reports.csv")


# # BLEU
all_reports_clean = all_reports.dropna()
ground_truth_tokens = [[ref.split()] for ref in list(all_reports_clean.content_findings)]
prediction_tokens = [pred.split() for pred in list(all_reports_clean.generated_report)]
smoothing = SmoothingFunction().method1
weights_unigram = (1.0,)
weights_bigram = (0.5, 0.5)
weights_trigram = (1/3, 1/3, 1/3)
weights_quadgram = (0.25, 0.25, 0.25, 0.25)
bleu_scores_weighted_unigram = [
    sentence_bleu(ref, pred, weights = weights_unigram, smoothing_function=smoothing) for ref, pred in zip(ground_truth_tokens, prediction_tokens)
]

average_bleu_score_unigram = sum(bleu_scores_weighted_unigram) / len(bleu_scores_weighted_unigram)

bleu_scores_weighted_bigram = [
    sentence_bleu(ref, pred, weights = weights_bigram, smoothing_function=smoothing) for ref, pred in zip(ground_truth_tokens, prediction_tokens)
]
average_bleu_score_bigram = sum(bleu_scores_weighted_bigram) / len(bleu_scores_weighted_bigram)
bleu_scores_weighted_trigram = [
    sentence_bleu(ref, pred, weights = weights_trigram, smoothing_function=smoothing) for ref, pred in zip(ground_truth_tokens, prediction_tokens)
]
average_bleu_score_trigram = sum(bleu_scores_weighted_trigram) / len(bleu_scores_weighted_trigram)

bleu_scores_weighted_quadgram = [
    sentence_bleu(ref, pred, weights = weights_quadgram, smoothing_function=smoothing) for ref, pred in zip(ground_truth_tokens, prediction_tokens)
]
average_bleu_score_quadgram = sum(bleu_scores_weighted_quadgram) / len(bleu_scores_weighted_quadgram)


# # BERTScore
from datasets import load_dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, EvalPrediction
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer, util

generated_reports = all_reports_clean['generated_report'].tolist()
content_findings = all_reports_clean['content_findings'].tolist()

precision, recall, f1 = score(generated_reports, content_findings, lang='en', verbose=True)

average_precision = precision.mean().item()
average_recall = recall.mean().item()
average_f1 = f1.mean().item()

#alternative approach for comparison 
bertscore = load("bertscore")
predictions = all_reports_clean.generated_report
references = all_reports_clean.content_findings
results = bertscore.compute(predictions=predictions, references=references, model_type="distilbert-base-uncased")

