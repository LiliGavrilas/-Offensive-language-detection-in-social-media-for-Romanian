import csv
import mysql.connector

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="1234",
  database="dex1"
)

mycursor = mydb.cursor()

mycursor.execute("SELECT lexicon FROM definition WHERE internalRep LIKE '%depr.%'")

myresult = mycursor.fetchall()

with open('bad_words_file.csv', mode='a+', newline='', encoding='utf-8') as employee_file:
  employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
  for x in myresult:
      employee_writer.writerow([x[0]])
