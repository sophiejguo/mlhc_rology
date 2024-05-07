AFS_ROOT_PATH = "/afs/csail/u/sophiejg/public"
# Model checkpoints
CHEXBERT_PATH = AFS_ROOT_PATH + "/models/chexbert.pth"
RADGRAPH_PATH = "radgraph/physionet.org/files/radgraph/1.0.0/models/model_checkpoint/model.tar.gz"

# Report paths
GT_REPORTS = AFS_ROOT_PATH + "/reports/gt_reports.csv"
PREDICTED_REPORTS = AFS_ROOT_PATH + "/reports/generated_reports.csv"
OUT_FILE = AFS_ROOT_PATH + "/reports/report_scores.csv"

# Whether to use inverse document frequency (idf) for BERTScore
USE_IDF = False
