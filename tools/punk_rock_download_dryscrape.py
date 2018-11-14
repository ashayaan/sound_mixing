import dryscrape
from bs4 import BeautifulSoup
import time
from tqdm import tqdm
import os
import requests


headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
def download_file(link, save_file_path):
    """
    takes in a URL and the output file save path downloads said file chunk by chunk with a progress bar to display
    :param link: link to download
    :param save_file_path: path to save the file in
    :return:
    """
    r = requests.get(link, headers=headers, stream=True)
    # file_size = int(r.headers['Content-Length'])
    chunk_size = 1024
    num_bars = 100

    with open(save_file_path, "wb") as code:
        for chunk in tqdm(r.iter_content(chunk_size=chunk_size), total=num_bars, unit='KB', desc=save_file_path,leave=True):
            code.write(chunk)


def download_songs(save_dir):
    save_dir_abs = os.path.abspath(save_dir)
    print save_dir_abs
    if not os.path.exists(save_dir_abs):
        os.makedirs(save_dir_abs)

    try:
        session = dryscrape.Session()
        session.visit("http://cambridge-mt.com/ms-mtk.htm#Rock")
        print "Waiting to render JavaScript"
        time.sleep(5)
        response = session.body()
    except Exception as e:
        import traceback
        print traceback.format_exc()
        raise e

    soup = BeautifulSoup(response,"lxml")

    flag = 0
    count = 1
    for song in soup.find_all('li'):
        artist = song.find('span',class_='artistHeader')

        if artist is None:
            continue
        else:
            artist_name = artist.text
            if artist_name == "All Hands Lost: 'Ambitions'":
                flag = 1

        if flag != 1:
            continue

        for song_info in song.find_all('li'):
            if song_info.find('span',class_ = 'projectNameType').find('strong',class_ = 'fullProjectTextColor') is None:
                continue
            datapoint_dir_name = ''.join(['song_', str(count)])
            datapoint_dir = os.path.join(save_dir_abs, datapoint_dir_name)

            if not os.path.exists(datapoint_dir):
                    os.makedirs(datapoint_dir)

            zip_html = song_info.find_all('span', class_='projectLine')[1]
            zip_file_link = zip_html.find('a', href=True)['href']
            zip_name = zip_file_link.split('/')[-1]
            save_file_path_zip = os.path.join(datapoint_dir, zip_name)

            mixed_file_link = song_info.find_all('span', class_='projectLine')[0].find('a', href=True)['href']
            mixed_file_name = mixed_file_link.split('/')[-1]
            save_file_path_mixed = os.path.join(datapoint_dir, mixed_file_name)

            name = zip_name.split('.zip')[0]

            print zip_file_link

            print "----------------------------------------------------------------------------------------------------"
            print "{} FILE NAME : {}".format(count, name)
            print datapoint_dir

            try:
                download_file(mixed_file_link, save_file_path_mixed)
                download_file(zip_file_link, save_file_path_zip)
            except Exception as e:
                import traceback
                error = traceback.format_exc()
                print "FAILED -- FILE NAME :", name
                print error
            time.sleep(1)
            count += 1


if __name__ == '__main__':
    save_dir = '../dataset_raw'
    download_songs(save_dir)