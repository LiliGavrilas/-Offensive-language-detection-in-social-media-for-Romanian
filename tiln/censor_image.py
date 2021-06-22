'''
import time
import urllib.request
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

driver = webdriver.Chrome()
wait = WebDriverWait(driver,10)
driver.get("https://www.kuaikanmanhua.com/web/comic/181959/")


for item in range(1):  # by increasing the highest range you can get more content
    wait.until(EC.visibility_of_element_located((By.TAG_NAME, "body"))).send_keys(Keys.END)
    time.sleep(80)
i = 0
for comment in wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".imgList img"))):
    urllib.request.urlretrieve(str(comment.get_attribute("data-src"))
                               , f"C:/Users/Florentin/Desktop/TILN/test/GetManhuaProject/images/local{i}.jpg")
    i += 1


'''
import numpy

'''
from PIL import Image
from os import walk
import cv2
import numpy as np
import re

f = []
for (dirpath, dirnames, filenames) in walk("C:/Users/Florentin/Desktop/TILN/test/GetManhuaProject/images"):
    f.extend(filenames)
    break

def sortFunc(e):
    return int(re.search(r"local([0-9]+).jpg", e).group(1))
f.sort(key=sortFunc)

def cut_line(path):
    img = cv2.imread(path)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((7, 7), np.uint8)
    erosion = cv2.erode(imgGray, kernel, iterations=1)
    hist = np.mean(erosion, axis=1)
    h = np.array([k[0] for (k, v) in np.ndenumerate(hist) if v == 2.55000000e+02])
    if len(h) == 0: return -1
    (_, _, _, ln) = max([(len(l), l[0], l[len(l) - 1], (l[len(l) - 1] - l[0]) // 2) for l in consecutive(h)])
    return ln

imgs = [Image.open(f"C:/Users/Florentin/Desktop/TILN/test/GetManhuaProject/images/{i}") for i in f if sortFunc(i) < len(f)/4]

# If you're using an older version of Pillow, you might have to use .size[0] instead of .width
# and later on, .size[1] instead of .height
min_img_width = min(i.width for i in imgs)

total_height = 0
for i, img in enumerate(imgs):
    # If the image is larger than the minimum width, resize it
    if img.width > min_img_width:
        imgs[i] = img.resize((min_img_width, int(img.height / img.width * min_img_width)), Image.ANTIALIAS)
    total_height += imgs[i].height

# I have picked the mode of the first image to be generic. You may have other ideas
# Now that we know the total height of all of the resized images, we know the height of our final image
img_merge = Image.new(imgs[0].mode, (min_img_width, total_height))
y = 0
for img in imgs:
    img_merge.paste(img, (0, y))

    y += img.height

for img in imgs:
    ln = cut_line(img)
    if ln == -1: continue
    img1 = img[0:ln, 0:640]
    img2 = img[ln:500, 500:640]
    img1.save(f"C:/Users/Florentin/Desktop/TILN/test/GetManhuaProject/images/{sortFunc(img)}_1.jpg")
    img1.save(f"C:/Users/Florentin/Desktop/TILN/test/GetManhuaProject/images/{sortFunc(img)}_2.jpg")

img_merge.save('cap1.jpg')
'''
# from colab_interface import test_comment
# import easyocr
# import cv2
# from matplotlib import pyplot as plt
# import numpy as np
#
# img = "C:/Users/ana_p/PycharmProjects/tiln/bad_photo3.jpg"
#
# reader = easyocr.Reader(['ro'], gpu=True)
# result = reader.readtext(img)
# print(result)

'''
try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract
from unidecode import unidecode

def ocr_core(filename):
    """
    This function will handle the core OCR processing of images.
    """
    text =pytesseract.image_to_string(Image.open(filename), lang="chi_sim")  # We'll use Pillow's Image class to open the image and pytesseract to detect the string in the image
    return text

print(ocr_core('images/local11.jpg'))
'''

'''
from colab_interface import test_comment
import easyocr
import cv2
from matplotlib import pyplot as plt
import numpy as np

img = "C:/Users/ana_p/PycharmProjects/tiln/bad_photo3.jpg"

reader = easyocr.Reader(['ro'], gpu=True)
result = reader.readtext(img)
print(result)

import pytesseract
import cv2
from colab_interface import test_comment


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def consorRoi(roi):
    w, h = (16, 16)
    height, width = roi.shape[:2]
    blur = cv2.GaussianBlur(roi, (7, 7), 1)
    temp = cv2.resize(blur, (w, h), interpolation=cv2.INTER_LINEAR)
    output = cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)
    return output

image = cv2.imread('C:/Users/ana_p/PycharmProjects/tiln/bad_photo3.jpg', 0)
thresh = 255 - cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

comment = ""
model = "svm_model"

for roi in result:
    x, y, w, h = roi[0][0][0], roi[0][0][1], abs(roi[0][1][0]-roi[0][0][0]), abs(roi[0][2][1]-roi[0][1][1])
    ROI = image[y:y+h, x:x+w]
    data = pytesseract.image_to_string(ROI, lang='ron',config='--psm 6')
    #image[y:y+h, x:x+w] = consorRoi(ROI)
    comment += data + " "
    print(data)

output, impact_words = test_comment(comment, model)
print(output)
if output == "Output: Offensive":
    for roi in result:
        x, y, w, h = roi[0][0][0], roi[0][0][1], abs(roi[0][1][0] - roi[0][0][0]), abs(roi[0][2][1] - roi[0][1][1])
        ROI = image[y:y + h, x:x + w]
        data = pytesseract.image_to_string(ROI, lang='ron', config='--psm 6')
        image[y:y+h, x:x+w] = consorRoi(ROI)
        #comment += data + " "
        #print(data)


cv2.imshow("Output", image)
cv2.waitKey(0)
'''
from colab_interface import test_comment
import easyocr
import cv2
from matplotlib import pyplot as plt
import numpy as np
import pytesseract
import cv2
from colab_interface import test_comment


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def consorRoi(roi):
    w, h = (16, 16)
    height, width = roi.shape[:2]
    blur = cv2.GaussianBlur(roi, (7, 7), 1)
    temp = cv2.resize(blur, (w, h), interpolation=cv2.INTER_LINEAR)
    output = cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)
    return output

def test_image(img, model):
    img = cv2.imdecode(numpy.fromstring(img.read(), numpy.uint8), cv2.IMREAD_UNCHANGED)
    reader = easyocr.Reader(['ro'], gpu=True)
    result = reader.readtext(img)
    #print(result)

    comment = ""

    for roi in result:
        x, y, w, h = int(round(roi[0][0][0])), int(round(roi[0][0][1])), int(round(abs(roi[0][1][0] - roi[0][0][0]))), int(round(abs(roi[0][2][1] - roi[0][1][1])))
        ROI = img[y:y + h, x:x + w]
        data = pytesseract.image_to_string(ROI, lang='ron', config='--psm 6')
        # image[y:y+h, x:x+w] = consorRoi(ROI)
        comment += data + " "
        print(data)

    output, impact_words = test_comment(comment, model)
    print(output)
    if output == "Output: Offensive":
        for roi in result:
            x, y, w, h = int(round(roi[0][0][0])), int(round(roi[0][0][1])), int(
                round(abs(roi[0][1][0] - roi[0][0][0]))), int(round(abs(roi[0][2][1] - roi[0][1][1])))
            ROI = img[y:y + h, x:x + w]
            data = pytesseract.image_to_string(ROI, lang='ron', config='--psm 6')
            img[y:y + h, x:x + w] = consorRoi(ROI)
            # comment += data + " "
            # print(data)

    return output, img





#([[55, 0], [147, 0], [147, 241], [55, 241]], '譬', 0.0004989605779749523)


'''
import pytesseract
import cv2


def transcribe(path, x1, y1, x2, y2):
    image = cv2.imread(path, 0)
    thresh = 255 - cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    x,y,w,h = x1, y1, abs(x2-x1), abs(y2-y1)
    ROI = thresh[x:y + h, x:x + w]
    data = pytesseract.image_to_string(ROI, lang='chi_sim', config='--psm 6')
    print(data)
'''
