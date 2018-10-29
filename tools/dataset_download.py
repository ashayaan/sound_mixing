import os
import requests

from tqdm import tqdm
from bs4 import BeautifulSoup

headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}


def download_file(link, save_file_path):
    """
    takes in a URL and the output file save path downloads said file chunk by chunk with a progress bar to display
    :param link: link to download
    :param save_file_path: path to save the file in
    :return:
    """
    r = requests.get(link, headers=headers, stream=True)
    file_size = int(r.headers['Content-Length'])
    chunk_size = 1024
    num_bars = int(file_size / chunk_size)

    with open(save_file_path, "wb") as code:
        for chunk in tqdm(r.iter_content(chunk_size=chunk_size), total=num_bars, unit='KB', desc=save_file_path,
                          leave=True):
            code.write(chunk)


def download_songs_from_cambridge_mt(download_url, save_dir):
    """
    This function downloads the songs from cambridge_mt - Practice Audio Files from the book - Mixing Secrets by Mike Senior
    :param download_url: the url to download the songs from
    :param save_dir: directory of dataset_raw - subdirectories will be created for every song
    :return:
    """
    # create save_dir if path does not exist
    save_dir_abs = os.path.abspath(save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    page = requests.get(download_url)
    soup = BeautifulSoup(page.text, "lxml")

    count = 1
    for song in soup.find_all('li'):
        for full in song.find_all('li'):
            if full.find('span', class_='projectNameType').find('strong', class_='fullProjectTextColor') is None:
                continue

            datapoint_dir_name = ''.join(['song_', str(count)])
            datapoint_dir = os.path.join(save_dir_abs, datapoint_dir_name)
            if not os.path.exists(datapoint_dir):
                os.makedirs(datapoint_dir)

            zip_html = full.find_all('span', class_='projectLine')[1]
            zip_file_link = zip_html.find('a', href=True)['href']
            zip_name = zip_file_link.split('/')[-1]
            save_file_path_zip = os.path.join(datapoint_dir, zip_name)

            mixed_file_link = full.find_all('span', class_='projectLine')[0].find('a', href=True)['href']
            mixed_file_name = mixed_file_link.split('/')[-1]
            save_file_path_mixed = os.path.join(datapoint_dir, mixed_file_name)

            name = zip_name.split('.zip')[0]
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
            count += 1


if __name__ == '__main__':
    cambridge_urls = ['http://cambridge-mt.com/ms-mtk.htm#AltRock', 'http://cambridge-mt.com/ms-mtk.htm#Rock']
    save_dir = '../dataset_raw'
    for download_url in cambridge_urls:
        download_songs_from_cambridge_mt(download_url, save_dir)