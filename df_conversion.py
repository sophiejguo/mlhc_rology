# convert dataframes to report format needed for CXR-Report-Metric

import pandas as pd
import pdb

PATH = '/om/user/sophiejg/project/mlhc_rology/'

generated_report_path = PATH + 'rology_cheXagent/logs/cheXagent_custom_generated_reports.csv'
gt_report_path = PATH + 'rology_CXR_LLaVA/dataset/rology_batch1.csv'


def subset_df(df, report_column):
    sub_df = pd.DataFrame()
    sub_df['study_id'] = df['Study_id']
    sub_df['report'] = df[report_column]
    sub_df = sub_df[sub_df['report'].notnull()]
    sub_df['report'] = sub_df['report'].astype(str)
    return sub_df


subset_gen_rep = subset_df(pd.read_csv(generated_report_path), 'generated_report')
subset_gen_rep.to_csv(PATH+'CXR-Report-Metric/reports/cheXagent_custom_generated_reports.csv')

subset_gt_rep = subset_df(pd.read_csv(gt_report_path), 'content_findings')
subset_gt_rep.to_csv(PATH+'CXR-Report-Metric/reports/gt_reports.csv')
 




