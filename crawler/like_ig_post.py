import requests
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from time import *

url = 'https://www.instagram.com/liu_din_ken/'
driver = webdriver.Chrome()
driver.get(url)

with open('Cookie.json') as f:
    cookies = json.load(f)

for cookie in cookies:
    driver.add_cookie(cookie)

sleep(1)
driver.refresh()
sleep(1)
action = ActionChains(driver)

eles = driver.find_elements(By.CLASS_NAME, '_ac7v.xras4av.xgc1b0m.xat24cr.xzboxd6')
for n in range(1):
    for ele in eles:
        sleep(0.2)
        like = None
        dislike = None
        switch = None

        click_eles = ele.find_elements(By.CSS_SELECTOR, '.x1lliihq.x1n2onr6.xh8yej3.x4gyw5p.xfllauq.xo2y696.x11i5rnm.x2pgyrj')
        for click_ele in click_eles:
            action.move_to_element(click_ele).click().perform()
            sleep(0.5)

    # 全部按讚把swith改成0，全部dislike改成1，like=0、dislike=1是交替
            try:
                like = driver.find_element(By.CSS_SELECTOR, '.x1lliihq.x1n2onr6.xyb1xck')
                switch = 0
            except:
                like = None
            try:
                dislike = driver.find_element(By.CSS_SELECTOR, '.x1lliihq.x1n2onr6.xxk16z8')
                switch = 1
            except:
                dislike = None

            if switch == 0:
                try:
                    action.move_to_element(like).click().perform()
                    sleep(0.5)
                except:
                    pass
            elif switch == 1:
                try:
                    action.move_to_element(dislike).click().perform()
                    sleep(0.5)
                except:
                    pass

            close = driver.find_element(By.CSS_SELECTOR, '.x1lliihq.x1n2onr6.x9bdzbf')
            action.move_to_element(close).click().perform()