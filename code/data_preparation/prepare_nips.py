__author__ = 'mdenil'

import requests
import os
import re
import sh
import psutil
import ruffus
import simplejson as json
import pyprind
import codecs
import urlparse
import tempfile
from bs4 import BeautifulSoup
from collections import Counter
from nltk.tokenize import WordPunctTokenizer


data_dir = "../data"
papers_dir = os.path.join(data_dir, "nips_papers")


def get_abstract(soup):
    return soup.find('p', class_="abstract").text


def get_paper_content(root, pdf_url):
    pdf_url = urlparse.urljoin(root, pdf_url)
    req = requests.get(pdf_url)

    pdf_file, pdf_file_name = tempfile.mkstemp(suffix=".pdf")
    pdf_file = os.fdopen(pdf_file, 'wb')
    pdf_file.write(req.content)
    pdf_file.close()
    sh.pdftotext(pdf_file_name)
    text_file_name = re.sub("pdf$", "txt", pdf_file_name)

    with codecs.open(text_file_name, encoding='utf-8') as text_file:
        text = text_file.read()

    os.remove(pdf_file_name)
    os.remove(text_file_name)

    return text


def get_paper_info_from_paper_page(root, paper_page_url):
    paper_id = re.match("^/paper/([0-9]+)", paper_page_url).group(1)

    paper_page_url = urlparse.urljoin(root, paper_page_url)

    req = requests.get(paper_page_url)
    soup = BeautifulSoup(req.text)

    pdf_url = soup.find('a', text="[PDF]").get("href")

    content = get_paper_content(root, pdf_url)
    abstract = get_abstract(soup)

    return {
        'content': content,
        'abstract': abstract,
        'url': paper_page_url,
        'paper_id': int(paper_id),
    }


def get_paper_urls_from_archive_page(root, archive_page_url):
    req = requests.get(urlparse.urljoin(root, archive_page_url))
    soup = BeautifulSoup(req.text)
    paper_links = soup.find_all('a', href=re.compile("^/paper/"))
    return [link.get("href") for link in paper_links]


@ruffus.originate(["nips_papers.json"])
def download_nips(output_file):
    nips_website_root = "http://papers.nips.cc/"

    nips_archive_urls = [
        "book/advances-in-neural-information-processing-systems-26-2013",
        "book/advances-in-neural-information-processing-systems-25-2012",
        "book/advances-in-neural-information-processing-systems-24-2011",
        "book/advances-in-neural-information-processing-systems-23-2010",
        "book/advances-in-neural-information-processing-systems-22-2009",
        "book/advances-in-neural-information-processing-systems-21-2008",
        "book/advances-in-neural-information-processing-systems-20-2007",
    ]

    paper_urls = []
    for url in nips_archive_urls:
        paper_urls.extend(
            get_paper_urls_from_archive_page(nips_website_root, url))

    progress_bar = pyprind.ProgPercent(len(paper_urls))
    with codecs.open(output_file, 'wb', encoding='utf-8') as nips_papers_file:
        for paper_url in paper_urls:
            paper_info = get_paper_info_from_paper_page(nips_website_root, paper_url)
            nips_papers_file.write("{}\n".format(json.dumps(paper_info)))
            progress_bar.update()


    # read and re-write so we actually have a file with single list instead of a sequence of
    # json dicts.  We didn't do this originally because we want to be able to write partial result
    # files in case something goes wrong mid-download
    with codecs.open(output_file, encoding='utf-8') as nips_papers_file:
        paper_infos = [json.loads(s) for s in nips_papers_file]

    with codecs.open(output_file, 'wb', encoding='utf-8') as nips_papers_file:
        nips_papers_file.write(u"{}\n".format(json.dumps(paper_infos)))


@ruffus.transform(download_nips, ruffus.suffix(".json"), ".clean.json")
def clean_data(input_file_name, output_file_name):
    def clean_word(word):
        word = word.lower()
        return word

    data = []
    with open(input_file_name) as input_file:
        for record in json.loads(input_file.read()):
            record['content'] = " ".join(map(clean_word, record['content'].split()))
            record['abstract'] = " ".join(map(clean_word, record['abstract'].split()))
            data.append(record)

    with codecs.open(output_file_name, 'w', encoding='utf-8') as output_file:
        json.dump(data, output_file)


@ruffus.transform(
    [clean_data],
    ruffus.suffix(".json"), ".dictionary.json")
def build_word_dictionary(input_file_name, output_file_name):
    dictionary = Counter()
    tokenizer = WordPunctTokenizer()
    with open(input_file_name) as input_file:
        for record in json.loads(input_file.read()):
            dictionary.update(tokenizer.tokenize(record['content']))
            dictionary.update(tokenizer.tokenize(record['abstract']))

    dictionary = list(sorted(w for w in dictionary if dictionary[w] >= 5)) + ['PADDING', 'UNKNOWN']

    with open(output_file_name, 'w') as output_file:
        output_file.write("{}\n".format(json.dumps(dictionary)))


if __name__ == "__main__":
    if not os.path.exists(papers_dir):
        os.makedirs(papers_dir)

    sh.cd(papers_dir)
    ruffus.pipeline_run(verbose=3, multiprocess=psutil.NUM_CPUS)