"""Download files with progress bar."""
import os
import re
import tarfile 
import hashlib
import requests
import argparse
from tqdm import tqdm
from constants import DOWNLOAD_INFO, DATA_PATH, str2bool

def parse_args():
    parser = argparse.ArgumentParser(
        description='Dataset downloader.',
        epilog='Example: python download.py --dataset voc2012',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--dataset', type=str, default='voc2012', help='Dataset to download')
    parser.add_argument('--store-dir', type=str, default=DATA_PATH, help='Path to a directory that store dataset')
    parser.add_argument('--keep', type=str2bool, default=True, help='Keep compressed file after extracting')
    args = parser.parse_args()
    return args


def check_sha1(filename, sha1_hash):
    """Check whether the sha1 hash of the file content matches the expected hash.

    ----------
    Args:

    filename : str
        Path to the file.
    sha1_hash : str
        Expected sha1 hash in hexadecimal digits.

    -------
    Return
    bool
        Whether the file content matches the expected hash.
    """
    sha1 = hashlib.sha1()
    with open(filename, 'rb') as f:
        while True:
            data = f.read(1048576)
            if not data:
                break
            sha1.update(data)

    sha1_file = sha1.hexdigest()
    l = min(len(sha1_file), len(sha1_hash))
    return sha1.hexdigest()[0:l] == sha1_hash[0:l]

def is_valid_url(url):
    regex = re.compile(
        r'^(?:http|ftp)s?://' # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' #domain...
        r'localhost|' #localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
        r'(?::\d+)?' # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return re.match(regex, url) is not None

def download_url(url, path=None, overwrite=False, sha1_hash=None):
    """Download an given URL

    ----------
    Args:

    url : str
        URL to download
    path : str, optional
        Destination path to store downloaded file. By default stores to the
        current directory with same name as in url.
    overwrite : bool, optional
        Whether to overwrite destination file if already exists.
    sha1_hash : str, optional
        Expected sha1 hash in hexadecimal digits. Will ignore existing file when hash is specified
        but doesn't match.
    
    -------
    Return
    str
        The file path of the downloaded file.
    """
    if not is_valid_url(url):
        raise 'URL is invalid!'
    if path is None:
        fname = url.split('/')[-1]
    else:
        path = os.path.expanduser(path)
        if os.path.isdir(path):
            fname = os.path.join(path, url.split('/')[-1])
        else:
            fname = path

    if overwrite or not os.path.exists(fname) or (sha1_hash and not check_sha1(fname, sha1_hash)):
        dirname = os.path.dirname(os.path.abspath(os.path.expanduser(fname)))
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        print('Downloading %s from %s...'%(fname, url))
        r = requests.get(url, stream=True)
        if r.status_code != 200:
            raise RuntimeError("Failed downloading url %s"%url)
        total_length = r.headers.get('content-length')
        with open(fname, 'wb') as f:
            if total_length is None: # no content length header
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk: # filter out keep-alive new chunks
                        f.write(chunk)
            else:
                total_length = int(total_length)
                for chunk in tqdm(
                    r.iter_content(chunk_size=1024), total=int(total_length / 1024. + 0.5), 
                    unit='KB', unit_scale=False, dynamic_ncols=True):
                    
                    f.write(chunk)

        if sha1_hash and not check_sha1(fname, sha1_hash):
            raise UserWarning('File {} is downloaded but the content hash does not match. ' \
                              'The repo may be outdated or download may be incomplete. ' \
                              'If the "repo_url" is overridden, consider switching to ' \
                              'the default repo.'.format(fname))

    return fname

def extractor(file, path='./', keep=False):
    """Extracting compressed file

    -------
    Args:

    file : str
        Path to compressed file
    path : str, optional
        Destination path to store decompressed file. By default stores to the
        current directory with same name as in url.
    keep : bool, optional
        Whether to keep compressed file after extracting
    """
    print(f"Extracting {file} ...")
    with tarfile.open(file) as tar:
        for member in tqdm(iterable=tar.getmembers(), total=len(tar.getmembers())):
            tar.extract(member=member, path=path)
    if not keep:
        os.remove(file)

if __name__ == '__main__':
    args = parse_args()
    file = download_url(DOWNLOAD_INFO[args.dataset]['url'], args.store_dir, DOWNLOAD_INFO[args.dataset]['hash'])
    extractor('./data/VOCtrainval_11-May-2012.tar', args.store_dir, args.keep)