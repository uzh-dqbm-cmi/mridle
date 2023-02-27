import smtplib
import datetime
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formatdate
import configparser

# Read the configuration file
config = configparser.ConfigParser()
config.read('/home/USZ/mcmamacc/config.ini')

# Access the values in the configuration file
username = config['DEFAULT']['username']
password = config['DEFAULT']['password']
recipients = ['mark.mcmahon@uzh.ch', 'markronan.mcmahon@usz.ch']  # config['DEFAULT']['recipients']

# create an SMTP object
smtp_obj = smtplib.SMTP('outlook.usz.ch', 587)

# establish a secure connection
smtp_obj.starttls()

# login to the email server using your email address and password
smtp_obj.login(username, password)
print(username)
print(recipients)
# create the email message
msg = MIMEMultipart()
msg['From'] = username
msg['To'] = ", ".join(recipients)
msg['Date'] = formatdate(localtime=True)
msg['Subject'] = 'DataFrame as csv attachment'
body = "Time sent {}".format(datetime.datetime.now())
msg.attach(MIMEText(body, 'plain'))

# Add the CSV attachment to the email
with open('/data/mridle/data/silent_live_test/my_dataframe.csv', 'rb') as csv_file:
    csv_attachment = MIMEApplication(csv_file.read(), _subtype='csv')
    csv_attachment.add_header('Content-Disposition', 'attachment', filename='my_dataframe.csv')
    msg.attach(csv_attachment)


# send the email
smtp_obj.sendmail(username, recipients, msg.as_string())

# close the SMTP connection
smtp_obj.quit()
