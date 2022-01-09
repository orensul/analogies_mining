
from collections import namedtuple
import os.path
import urllib.request
import sys
import time
import tarfile
from textwrap import TextWrapper

# copied from:
# https://blog.shichao.io/2012/10/04/progress_speed_indicator_for_urlretrieve_in_python.html
def show_progress(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write("\r%d%%, %d MB, %d KB/s, %d seconds passed" %
                    (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()

Download = namedtuple('Download', ['name', 'path', 'url', 'ext', 'description', 'should_untar'])

wrapper = TextWrapper(width=80, initial_indent="     ", subsequent_indent="     ")

downloads = [
    Download(
        name = 'span model (binary classification)',
        path = 'models/span_binary.tar.gz',
        url = 'https://www.dropbox.com/s/jpu992r4umvlnq3/span_binary.tar.gz?dl=1',
        ext = '',
        description = "Span detector model matching the design of that in FitzGerald et al. (2018)."
        " Performs binary classification over each span, trained to predict whether the answer was provided"
        " for a predicate by any annotator.",
        should_untar = False,
    ),
    Download(
        name = 'span model (probability density)',
        path = 'models/span_density_softmax.tar.gz',
        url = 'https://www.dropbox.com/s/1pmid0cdrp49efv/span_density_softmax.tar.gz?dl=1',
        ext = '',
        description = "Span detector model which is different from that of FitzGerald et al. (2018) in that"
        " it estimates a probability distribution over answer spans (matching the annotated distribution)"
        " rather that acting as a binary classifier. This doesn't work quite as well on raw metrics like F1"
        " but could be useful in some circumstances.",
        should_untar = False,
    ),
    Download(
        name = 'span -> question model',
        path = 'models/span_to_question.tar.gz',
        url = 'https://www.dropbox.com/s/monbzb3afkmo3j5/span_to_question.tar.gz?dl=1',
        ext = '',
        description = "QA-SRL question generator that conditions on an answer span.",
        should_untar = False,
    ),
    Download(
        name = 'span -> simplified question model',
        path = 'models/span_to_simplified_question.tar.gz',
        url = 'https://www.dropbox.com/s/rvuwcl9kedsb2z4/span_to_simplified_question.tar.gz?dl=1',
        ext = '',
        description = "Question generator conditioning on answer spans and producing"
        " simplified questions with normalized tense, aspect, negation, modality, and animacy."
        " Used in 'Inducing Semantic Roles Without Syntax' (https://github.com/julianmichael/qasrl-roles).",
        should_untar = False,
    ),
]

a = ord('a')

def get_download_option_prompt(num, download):
    if os.path.exists(download.path):
        color = "\u001b[32m"
        icon  = "[downloaded]"
    else:
        color = "\u001b[33m"
        icon  = "[not downloaded]"

    letter = chr(num + a)

    desc = ("\n" + "\n".join(wrapper.wrap(download.description)))

    return u"  {}) {}{} {}\u001b[0m ".format(letter, color, download.name, icon) + desc + "\n"


def construct_prompt():
    prompt = "What would you like to download? ('all' to download all, 'q' to quit)\n"
    for i, download in enumerate(downloads):
        prompt += "\n" + get_download_option_prompt(i, download)
    return prompt

def download_item(download):
    print("Downloading {}.".format(download.name))
    if len(download.ext) > 0 and download.should_untar:
       tarpath = download.path + download.ext
       urllib.request.urlretrieve(download.url, tarpath, show_progress)
       result = tarfile.open(tarpath)
       result.extractall(os.path.dirname(download.path))
       result.close()
       os.remove(tarpath)
    else:
       urllib.request.urlretrieve(download.url, download.path, show_progress)
    print("\nDownload complete: {}".format(download.path))

should_refresh_prompt = True

def download_items(spec):
    should_refresh_prompt = False
    print(spec)
    if spec.lower() in ["*", "all"]:
        for i, download in enumerate(downloads):
            print("{}) {}".format(chr(i + a), download.description))
            if os.path.exists(download.path):
                print("Already downloaded at {}.".format(download.path))
            else:
                download_item(download)
    else:
        for c in spec:
            try:
                choice = downloads[ord(c) - a]
            except ValueError or IndexError:
                print("Invalid option: {}".format(spec))
                continue
            if os.path.exists(choice.path):
                print("Already downloaded at {}.".format(choice.path))
                print("Re-download? [y/N] ", end='')
                shouldDownloadStr = input()
                if shouldDownloadStr.startswith("y") or \
                shouldDownloadStr.startswith("Y"):
                    download_item(choice)
                    should_refresh_prompt = True
            else:
                download_item(choice)
                should_refresh_prompt = True


if len(sys.argv) > 1:
    download_items(sys.argv[1])
else:
    while True:
        if should_refresh_prompt:
            print(construct_prompt())
        optstr = "".join([chr(i) for i in range(a, a + len(downloads))])
        print("Choose a subset ({}/all/q): ".format(optstr), end='')
        spec = input()
        if "quit".startswith(spec.lower()):
            break
        else:
            download_items(spec)
