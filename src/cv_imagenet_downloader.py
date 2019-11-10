#!/usr/bin/env python
# Copyright (c) 2019 Jose del Rio
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


import os
import sys
import math
import time
import click
# import argparse
import threading
import imghdr
import http.client
from nltk.corpus import wordnet as wn    # wordnet.synset_from_pos_and_offset(pos, offset)
from ssl import CertificateError
from socket import timeout as TimeoutError
from socket import error as SocketError
import urllib.request, urllib.error, urllib.parse


class DownloadError(Exception):
    """Base class for exceptions in this module."""
    def __init__(self, message=""):
        self.message = message


def download(url, timeout, retry, sleep=0.8):
    """Downloads a file at given URL."""
    count = 0
    while True:
        try:
            f = urllib.request.urlopen(url, timeout=timeout)
            if f is None:
                raise DownloadError('Cannot open URL' + url)
            content = f.read()
            f.close()
            break
        except (urllib.error.HTTPError, http.client.HTTPException, CertificateError) as err:
            count += 1
            if count > retry:
                raise DownloadError()
        except (urllib.error.URLError, TimeoutError, SocketError, IOError) as err:
            count += 1
            if count > retry:
                raise DownloadError('failed to open ' + url + ' after ' + str(retry) + ' retries')
            time.sleep(sleep)
        except Exception as err:
            print('otherwise uncaught error: ' + err)
    return content


def make_directory(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def get_url_request_list_function(request_url):
    def get_url_request_list(wnid, timeout=5, retry=3):
        url = request_url + wnid
        response = download(url, timeout, retry).decode()
        print("response: " + response)
        list = str.split(response)
        return list
    return get_url_request_list

get_image_urls = get_url_request_list_function('http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=')

get_subtree_wnid = get_url_request_list_function('http://www.image-net.org/api/text/wordnet.structure.hyponym?wnid=')

get_full_subtree_wnid = get_url_request_list_function('http://www.image-net.org/api/text/wordnet.structure.hyponym?full=1&wnid=')


def get_words_wnid(wnid, timeout=5, retry=3):
    url = 'http://www.image-net.org/api/text/wordnet.synset.getwords?wnid=' + wnid
    response = download(url, timeout, retry).decode()
    return response


def download_images(dir_path, image_url_list, n_images, min_size, timeout, retry, sleep):
    make_directory(dir_path)
    image_count = 0
    for url in image_url_list:
        if image_count >= n_images:
            break
        try:
            image = download(url, timeout, retry)
            try:
                extension = imghdr.what('', image)      # check if valid image
                if extension == "jpeg":
                    extension = "jpg"
                if extension is None:
                    raise DownloadError()
            except Exception:
                raise DownloadError()
            if sys.getsizeof(image) > min_size:
                image_name = "image_" + str(image_count) + '.' + extension
                image_path = os.path.join(dir_path, image_name)
                image_file = open(image_path, 'wb')
                image_file.write(image)
                image_file.close()
                image_count += 1
                time.sleep(sleep)
        except DownloadError as err:
            print(f"Could not download {url}. Error: {err}")


# Check info about WorNet IDs at https://wordnet.princeton.edu/
@click.command()
@click.option("-w", "--wnid", required=True, type=str, help="WordNet ID, ex n03592371")
@click.option("-o", "--outputdir", help="Output directory")
@click.option("-j", "--jobs", default=1, type=int, help="Number of parallel threads")
@click.option("-t", "--timeout", default=2, type=float, help="Timeout per request (seconds)")
@click.option("-r", "--retry", default=0, type=int, help="Number of retries per request")
@click.option("-s", "--sleep", default=0, type=int, help="Sleep after download each image (seconds)")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose messages")
@click.option("-i", "--n_images", default=50, type=int, help="Number of images per category")
@click.option("-f", "--fullsubtree", is_flag=True, help="Downloads the full subtree")
@click.option("-R", "--noroot", is_flag=True, help="Do not download the root")
@click.option("-S", "--nosubtree", is_flag=True, help="Do not download the subtree")
@click.option("-h", "--human_readable", is_flag=True, help="Use human readable folder names")
@click.option("-m", "--min_size", default=5000, type=float, help="Min size of the images (bytes)")
def main(wnid, outputdir, jobs, timeout, retry, n_images, sleep,
         verbose, fullsubtree, noroot, nosubtree, human_readable, min_size):

    wnids_list = []

    # First get the list of wnids
    if not noroot:
        wnids_list.append(wnid)
    if not nosubtree:
        if fullsubtree:
            subtree = get_full_subtree_wnid(wnid, timeout, retry)
        else:
            subtree = get_subtree_wnid(wnid, timeout, retry)
        for i in range(1, len(subtree)):
            # removes the dash
            subtree[i] = subtree[i][1:]
        wnids_list.extend(subtree)

    # create root directory
    make_directory(outputdir)
    # Split the wnids for the threads
    wnid_list_len = len(wnids_list)
    wnid_thread_lists = []
    wnid_thread_sizes = int(math.ceil(float(wnid_list_len) / jobs))
    print(wnid_list_len)
    for i in range(jobs):
        wnid_thread_lists.append(wnids_list[i * wnid_thread_sizes: (i+1) * wnid_thread_sizes])

    def downloader(wnid_list):
        # Define the threads
        for wnid in wnid_list:
            if human_readable:
                dir_name = get_words_wnid(wnid, timeout, retry)
            else:
                dir_name = wnid
            if verbose:
                print('Downloading ' + dir_name)
            dir_path = os.path.join(outputdir, dir_name)
            if os.path.isdir(dir_path):
                print('skipping: already have ' + dir_name)
            else:
                image_url_list = get_image_urls(wnid,timeout,retry)
                download_images(dir_path, image_url_list, n_images, min_size, timeout, retry, sleep)

    # initialize the threads
    print(wnid_thread_lists[0])
    download_threads = [threading.Thread(target=downloader, args=([wnid_thread_lists[i]])) for i in range(jobs)]

    try:
        for t in download_threads:
            t.start()
    except Exception:
        sys.exit(1)

    is_alive = True

    while is_alive:
        try:
            is_alive = False
            for t in download_threads:
                is_alive = is_alive or t.isAlive()
            sleep(0.1)
        except:
            sys.exit(1)

    try:
        for t in download_threads:
            t.join()
    except Exception:
        sys.exit(1)

    print("finished")


if __name__ == '__main__':
    # p = argparse.ArgumentParser()
    # p.add_argument('wnid', help='Imagenet wnid, example n03489162')
    # p.add_argument('outdir', help='Output directory')
    # p.add_argument('--jobs', '-j', type=int, default=1,
    #                help='Number of parallel threads to download')
    # p.add_argument('--timeout', '-t', type=float, default=2,
    #                help='Timeout per request in seconds')
    # p.add_argument('--retry', '-r', type=int, default=0,
    #                help='Max count of retry for each request')
    # p.add_argument('--sleep', '-s', type=float, default=0,
    #                help='Sleep after download each image in second')
    # p.add_argument('--verbose', '-v', action='store_true',
    #                help='Enable verbose messages')
    # p.add_argument('--images', '-i', type=int, default=20,
    #                metavar='N_IMAGES',
    #                 help= 'Number of images per category to download')
    # p.add_argument('--fullsubtree', '-F', action='store_true',
    #                help='Downloads the full subtree')
    # p.add_argument('--noroot', '-R', action='store_true',
    #                help='Do not Downloads the root')
    # p.add_argument('--nosubtree', '-S', action='store_true',
    #                help='Do not Downloads the subtree')
    #
    # p.add_argument('--humanreadable', '-H', action='store_true',
    #                help='Makes the folders human readable')
    #
    # p.add_argument('--minsize', '-m', type=float, default=7000,
    #                help='Min size of the images in bytes')

    # args = p.parse_args()
    # main(wnid=args.wnid,
    #      out_dir=args.outdir,
    #      jobs=args.jobs,
    #      timeout=args.timeout,
    #      retry=args.retry,
    #      n_images=args.images,
    #      sleep=args.sleep,
    #      verbose=args.verbose,
    #      fullsubtree=args.fullsubtree,
    #      noroot=args.noroot,
    #      nosubtree=args.nosubtree,
    #      human_readable=args.humanreadable,
    #      min_size=args.minsize)

    category = "dog"
    ss = wn.synsets(category, pos=wn.NOUN)[0]   # wn.synset('dog.n.01')
    offset = ss.pos() + str(ss.offset()).zfill(8)


    arguments = [
        "--wnid", offset,
        "--outputdir", category,
        "--jobs", 1,
    ]
    main(*arguments)

# Example:
# python -m cv_imagenet_downloader.py n03489162
