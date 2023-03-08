import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import smtplib
import datetime
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formatdate
import configparser

from mridle.pipelines.data_science.feature_engineering.nodes import add_business_days


def intervention(dt):
    """
    df: dataframe with appointments that need to be called for that day. Both intervention and control included . i.e.
    just the top 20 (or above threshold...). Should have a col called 'control' indicating if it is control or
    intervention.
    """

    filename_date = add_business_days(dt, 3).date().strftime('%Y_%m_%d')
    filename = '/data/mridle/data/silent_live_test/live_files/all/out_features_data/features_{}.csv'.format(
        filename_date)

    preds = pd.read_csv(filename, parse_dates=['start_time'])
    preds.rename(columns={"prediction_xgboost": "prediction"}, inplace=True)
    preds.drop(columns=[x for x in preds.columns if 'prediction_' in x], inplace=True)
    preds.drop(columns=[x for x in preds.columns if 'Unnamed:' in x], inplace=True)

    # Take the top X appts
    # preds = preds.sort_values("prediction", ascending=False)[:split_config[day_of_week_from_filename]['num_preds']]

    # Take appts above a certain threshold
    preds = preds[preds['prediction'] > 0.185]

    preds['control'] = 'control'

    # use the index of a sampling to change ~50% of the labels to 'intervention'
    preds.loc[preds.sample(frac=0.5, replace=False).index, 'control'] = 'intervention'

    intervention_df = preds[preds['control'] == 'intervention'][['MRNCmpdId', 'FillerOrderNo', 'start_time', 'Telefon']]

    # Save the original as csv, and then the intervention one as PDF to be emailed
    preds.to_csv("/data/mridle/data/intervention/intervention_{}.csv".format(filename_date), index=False)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')
    ax.table(cellText=intervention_df.values, colLabels=intervention_df.columns, loc='center')

    pp = PdfPages("/data/mridle/data/intervention/intervention_{}.pdf".format(filename_date))
    pp.savefig(fig, bbox_inches='tight')
    pp.close()

    # Read the configuration file
    config = configparser.ConfigParser()
    config.read('/data/mridle/data/intervention/config.ini')

    # Access the values in the configuration file
    username = config['DEFAULT']['username']
    password = config['DEFAULT']['password']
    recipients = config['DEFAULT']['recipients'].split(',')

    # create an SMTP object
    smtp_obj = smtplib.SMTP('outlook.usz.ch', 587)

    # establish a secure connection
    smtp_obj.starttls()

    # login to the email server using your email address and password
    smtp_obj.login(username, password)

    # create the email message
    msg = MIMEMultipart()
    msg['From'] = username
    msg['To'] = ", ".join(recipients)
    msg['Date'] = formatdate(localtime=True)
    msg['Subject'] = 'Intervention Study - {}'.format(filename_date)
    body = """
    Dear Namka,

    Here are the upcoming appointments which we would like to include in the study.

    As discussed, could you please:

    1. Try to call these patients today and remind them of their appointment
    2. Send me an email with some feedback (i.e. whether you could get talking with the patient, what they said, etc.)\
    in whichever form suits you.

    Let me know if you have any questions.

    Regards,
    Mark
    """
    msg.attach(MIMEText(body, 'plain'))

    path_to_pdf = '/data/mridle/data/intervention/intervention_{}.pdf'.format(filename_date)

    with open(path_to_pdf, "rb") as f:
        attach = MIMEApplication(f.read(), _subtype="pdf")
    attach.add_header('Content-Disposition', 'attachment', filename='Intervention_{}'.format(filename_date))
    msg.attach(attach)

    # send the email
    smtp_obj.sendmail(username, recipients, msg.as_string())

    # close the SMTP connection
    smtp_obj.quit()


if __name__ == '__main__':
    intervention(dt=datetime.datetime.today())
