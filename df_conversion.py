# convert dataframes to report format needed for CXR-Report-Metric

import pandas as pd

PATH = '/afs/csail/u/s/sophiejg/public/'

generated_report_path = PATH + 'generated_data/generated_reports.csv'
gt_report_path = PATH + 'Rology-dataset/rology_batch1.csv'


def subset_df(df, report_column):
    sub_df = pd.DataFrame()
    sub_df['subject_id'] = df['Study_id']
    sub_df['report'] = df[report_column]
    return sub_df


subset_gen_rep = subset_df(pd.read_csv(generated_report_path), 'generated_report')
subset_gen_rep.to_csv(PATH+'reports/generated_reports.csv')

subset_gt_rep = subset_df(pd.read_csv(gt_report_path), 'content_findings')
subset_gt_rep.to_csv(PATH+'reports/gt_reports.csv')




