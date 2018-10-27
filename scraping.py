import os
from bs4 import BeautifulSoup
import requests
import urllib

headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}


def downloadSongs(soup):
	count = 0
	for song in soup.find_all('li'):
		for full in song.find_all('li'):
			if full.find('span',class_='projectNameType').find('strong',class_='fullProjectTextColor') == None:
				continue
			x = full.find_all('span',class_='projectLine')[1]
			link =  x.find('a', href=True)['href']
			name = link.split('/')[-1]
			print name
			
			try:
				r = requests.get(link,headers=headers)
				with open(name, "wb") as code:
					code.write(r.content)
			except Exception as e:
				raise e
			
			# print name




if __name__ == '__main__':
	page = requests.get('http://cambridge-mt.com/ms-mtk.htm#PatrickTalbot_UpperHand')
	soup = BeautifulSoup(page.text,"lxml")
	downloadSongs(soup)