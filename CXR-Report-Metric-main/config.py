ROOT_PATH = "/om/user/sophiejg/project/mlhc_rology/CXR-Report-Metric"
# Model checkpoints
CHEXBERT_PATH = ROOT_PATH + "/models/chexbert.pth"
RADGRAPH_PATH = ROOT_PATH + "/models/model.tar.gz"

# Report paths
GT_REPORTS = ROOT_PATH + "/reports/gt_reports.csv"
PREDICTED_REPORTS = ROOT_PATH + "/reports/cheXagent_custom_generated_reports.csv"
OUT_FILE = ROOT_PATH + "/reports/cheXagent_report_scores.csv"

# Whether to use inverse document frequency (idf) for BERTScore
USE_IDF = False
