import os
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

driver = webdriver.Chrome()
driver.get('https://uma.komoejoy.com/character.html')
driver.implicitly_wait(10)

imgs = driver.find_elements(by = By.CSS_SELECTOR, value = 'img')
for img in imgs:
    try:
        url = 'http:' + img.get_attribute('data-src')
        img_data = requests.get(url).content
        filename = img.get_attribute('alt')
        os.makedirs('Documents/Github/desktop-tutorial/crawler/umamusume', exist_ok = True)
        with open(f'Documents/Github/desktop-tutorial/crawler/umamusume/{filename}.png', 'wb') as f:
            f.write(img_data)
    except:
        pass