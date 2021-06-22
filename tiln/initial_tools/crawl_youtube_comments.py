import csv
from _csv import reader

from bs4 import BeautifulSoup
"""
html_text = open("autosave.html", "r")
soup = BeautifulSoup(html_text, 'html.parser')

comm_list = soup.select('.text-fragment')

chapter = 1
image = 1

with open('comm_file.csv', mode='w', newline='') as employee_file:
    employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for comm in comm_list:
        if comm.text.strip() != "":
            employee_writer.writerow([comm.text.strip()])

for i in range(7, 13):
    html_text = open(f"autosave{i}.html", "r")
    soup = BeautifulSoup(html_text, 'html.parser')
    comm_list = soup.select('.text-fragment')
with open('comm_file.csv', mode='a+', newline='') as employee_file:
    employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for comm in comm_list:
        if comm.text.strip() != "":
            employee_writer.writerow([comm.text.strip()])
"""
'''
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

driver = webdriver.Chrome()
wait = WebDriverWait(driver,10)
driver.get("https://www.youtube.com/watch?v=C3yLFWyBHxM")


with open('comm_file3.csv', mode='a+', newline='', encoding='utf-8') as employee_file:
    employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for item in range(20):  # by increasing the highest range you can get more content
        wait.until(EC.visibility_of_element_located((By.TAG_NAME, "body"))).send_keys(Keys.END)
        time.sleep(3)

    for comment in wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "#comment #content-text"))):
            employee_writer.writerow([comment.text])
'''

'''
with open('comm_file2.csv', mode='r', newline='', encoding='cp1252') as employee_f:
    csv_reader = reader(employee_f)
    for comm in csv_reader:
        with open('comm_file_full.csv', mode='a+', newline='', encoding='utf-8') as employee_file:
            employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            separator = ''
            employee_writer.writerow([comm[0], separator.join(comm[1:])])
'''



# for img in image_list:
#     image_link = image_list.select('.img')
#
#         with open(f"{chapter}/{image}.png", "wb") as f:
#             f.write(requests.get(lnk).content)
