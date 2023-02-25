### 데이터 수집 및 저장(완료) ###
## 라이브러리 import
import os
import time
import lxml
import requests
import psycopg2
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from selenium import webdriver

# 환경 변수 가져오기
load_dotenv()
mozila = os.getenv('MOZILA')
psycopg_host = os.getenv('ELEPHANTSQL_HOST')
psycopg_db = os.getenv('ELEPHANTSQL_DB')
psycopg_user = os.getenv('ELEPHANTSQL_USER')
psycopg_password = os.getenv('ELEPHANTSQL_PASSWORD')

# 망고플레이트 스크레이핑 
def scrape_mongoplate(city):
    """
    해당 함수는 도시명을 받아 망고플레이트에 있는 식당의 정보를
    리스트로 리턴합니다. 
    - title : 식당이름, 문자열(str)
    - review_star : 별점(식당 평가), 실수(float)
    - menu : 판매 음식 종류(한식, 일식, 중식, 양식 등), 문자열(str)
    - view_count : 조회수, 정수(int)
    - review_count : 리뷰수, 정수(int)
    - city : 식당의 위치, 문자열(str)
    """
    store_list = []
    for i in range(10):
        page_url = f'https://www.mangoplate.com/en/search/{city}?keyword={city}&page={i}'
        page = requests.get(page_url, headers={'User-agent':mozila})
        soup = BeautifulSoup(page.content, 'html.parser')
        elements = soup.find_all('div', class_='info') 
        for j in range(len(elements)):
            title = elements[j].text.split(sep='\n')[2].replace("'","_")
            review_star = elements[j].text.split(sep='\n')[-7].replace("'","")
            loc_menu = elements[j].text.split(sep='\n')[-6].split(sep='-')[-1]
            view_count = elements[j].text.split(sep='\n')[-4].replace(',','')
            review_count = elements[j].text.split(sep='\n')[-3].replace(',','')
            di = {'title':title, 'review_star':review_star, 'menu':loc_menu, 'view_count':view_count, 'review_count':review_count, 'city':city}
            store_list.append(di)
        time.sleep(2)
    return store_list


# PostgreSQL에 DB생성
conn = psycopg2.connect(
    host = psycopg_host,
    database = psycopg_db,
    user = psycopg_user,
    password = psycopg_password
)
cur = conn.cursor()
cur.execute("DROP TABLE IF EXISTS store;")
cur.execute("""CREATE TABLE store (
    id SERIAL PRIMARY KEY,
    title VARCHAR(255),
    review_star VARCHAR(255),
    menu VARCHAR(255),
    view_count VARCHAR(255),
    review_count VARCHAR(255),
    city VARCHAR(255)
    );""")

# DB에 데이터 저장
city_list = ['강남구', '중구', '마포구', '종로구', '서초구', '송파구', '영등포구', '용산구', '광진구', '관악구', 
              '분당구', '남양주시', '부천시', '화성시','파주시','평택시','일산','수원','덕양',
              '해운대구','부산진구','수영구','동래구','기장군','부산중구',
              '강릉','춘천','원주','속초','수성구','대구중구','달서구','대구북구']
mongo_store_list = []
for row in city_list:
  a = scrape_mongoplate(row)
  mongo_store_list.append(a)
for i in range(len(mongo_store_list)):
    for j in range(len(mongo_store_list[i])):
        cur.execute(f"""INSERT INTO store (title, review_star, menu, view_count, review_count, city) 
        VALUES ('{mongo_store_list[i][j]['title']}','{mongo_store_list[i][j]['review_star']}',
        '{mongo_store_list[i][j]['menu']}','{mongo_store_list[i][j]['view_count']}',
        '{mongo_store_list[i][j]['review_count']}','{mongo_store_list[i][j]['city']}');""")
conn.commit()
conn.close()

### selenimu을 통한 네이버 쇼핑 정보 스크래핑
# chrome에서 네이버 쇼핑(키워드 밀키트) 열기
browser = webdriver.Chrome()
nav_list = []
for i in range(100):
    nav_url = f'https://search.shopping.naver.com/search/all?frm=NVSHTTL&origQuery=%EB%B0%80%ED%82%A4%ED%8A%B8&pagingIndex={i}&pagingSize=40&productSet=total&query=%EB%B0%80%ED%82%A4%ED%8A%B8&sort=rel&timestamp=&viewType=list'
    browser.get(nav_url)
    time.sleep(1)

    # 스크롤 내리기
    browser.execute_script('window.scrollTo(0,10000)')

    # 스크래핑
    nav_soup = BeautifulSoup(browser.page_source, 'html.parser')
    nav_items = nav_soup.find_all('div', attrs={'class':'basicList_info_area__TWvzp'})
    for nav_item in nav_items:
        nav_name = nav_item.find('a', attrs={'target':'_blank', 'class':'basicList_link__JLQJf'}).get_text()
        nav_price = nav_item.find('span', attrs={'class':'price_num__S2p_v','data-testid':'SEARCH_PRODUCT_PRICE'}).get_text()
        nav_fare = nav_item.find('span', attrs={'class':'price_delivery__yw_We'})
        if nav_fare:
            nav_fare = nav_fare.get_text()
        else:
            pass
        nav_cate = nav_item.find('div',attrs={'class':'basicList_depth__SbZWF'}).get_text()
        nav_etc = nav_item.find_all('a', attrs={'class':'basicList_etc__LSkN_'})
        for i in range(len(nav_etc)):
            if nav_etc[i]:
                nav_etc[i] = nav_etc[i].get_text()
            else :
                pass
        nav_etc = ' '.join(nav_etc)
        a = {'name' : nav_name, 'price':nav_price,'fare':nav_fare,'category':nav_cate,'etc':nav_etc}
        nav_list.append(a)
        time.sleep(1)
    time.sleep(2)

# PostgreSQL에 네이버 DB생성 및 저장
conn = psycopg2.connect(
    host = psycopg_host,
    database = psycopg_db,
    user = psycopg_user,
    password = psycopg_password
)
cur = conn.cursor()
cur.execute("DROP TABLE IF EXISTS naver;")
cur.execute("""CREATE TABLE naver (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255),
    price VARCHAR(255),
    fare VARCHAR(255),
    category VARCHAR(255),
    etc VARCHAR(255)
    );""")
for i in range(len(nav_list)):
    cur.execute(f"""INSERT INTO naver (name, price, fare, category, etc) 
    VALUES ('{nav_list[i]['name']}','{nav_list[i]['price']}',
    '{nav_list[i]['fare']}','{nav_list[i]['category']}','{nav_list[i]['etc']}');""")
conn.commit()
conn.close()