import requests
import random
import time


s = requests.Session()
random.seed(11)
while True:
    i = random.randrange(1, 29)
    if i == 1:
        req = s.get('https://www.youtube.com/', timeout=5, verify=False)
    if i == 2:
        req = s.get('https://www.billboard.com/', timeout=5, verify=False)
    if i == 3:
        req = s.get('https://www.google.com/', timeout=5, verify=False)
    if i == 4:
        req = s.get('https://www.daum.net/', timeout=5, verify=False)
    if i == 5:
        req = s.get('https://www.melon.com/', timeout=5, verify=False)
    if i == 6:
        req = s.get('https://www.genie.co.kr/', timeout=5, verify=False)
    if i == 7:
        req = s.get('https://www.instagram.com/', timeout=5, verify=False)
    if i == 8:
        req = s.get('https://www.facebook.com/', timeout=5, verify=False)
    if i == 9:
        req = s.get('https://www.naver.com/', timeout=5, verify=False)
    if i == 10:
        req = s.get('https://www.amazon.com/', timeout=5, verify=False)
    if i == 11:
        req = s.get('http://mju.ac.kr/', timeout=5, verify=False)
    if i == 12:
        req = s.get('https://www.wemakeprice.com/', timeout=5, verify=False)
    if i == 13:
        req = s.get('https://www.afreecatv.com/', timeout=5, verify=False)
    if i == 14:
        req = s.get('https://kr.leagueoflegends.com/', timeout=5, verify=False)
    if i == 15:
        req = s.get('https://www.kbstar.com/', timeout=5, verify=False)
    if i == 16:
        req = s.get('https://www.blizzard.com/', timeout=5, verify=False)
    if i == 17:
        req = s.get('https://www.netflix.com/', timeout=5, verify=False)
    if i == 18:
        req = s.get('https://www.sbs.co.kr/', timeout=5, verify=False)
    if i == 19:
        req = s.get('https://www.krafton.com/', timeout=5, verify=False)
    if i == 20:
        req = s.get('http://accs.mju.ac.kr/', timeout=5, verify=False)
    if i == 21:
        req = s.get('http://hmcl.mju.ac.kr/', timeout=5, verify=False)
    if i == 22:
        #req = s.get('https://cg.mju.ac.kr/', timeout=5, verify=False)
        req = s.get('http://www.fow.kr/', timeout=5, verify=False)
    if i == 23:
        req = s.get('http://www.fow.kr/', timeout=5, verify=False)
    if i == 24:
        req = s.get('http://torrentgaja.com/', timeout=5, verify=False)
    if i == 25:
        req = s.get('http://www.mlbpark.com/', timeout=5, verify=False)
    if i == 26:
        req = s.get('http://www.soribada.com/', timeout=5, verify=False)
    if i == 27:
        req = s.get('http://www.mnet.com/', timeout=5, verify=False)
    if i == 28:
        req = s.get('http://www.yes24.com/', timeout=5, verify=False)
    if i == 29:
        req = s.get('http://www.encar.com/', timeout=5, verify=False)
    print(i)
    j = random.randrange(1, 3)
    time.sleep(j)
    
    
    
    
    
    
    
    

