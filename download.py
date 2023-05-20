"""Download files with progress bar."""
import os
import re
import zipfile
import tarfile 
import hashlib
import requests
import argparse
from tqdm import tqdm
from utils import str2bool
from config import DATASET_INFO, DATA_PATH, CHUNK_SIZE

def parse_args():
    parser = argparse.ArgumentParser(
        description='Dataset downloader.',
        epilog='Example: python download.py --dataset voc2012',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--dataset', type=str, default='voc2012', choices=DATASET_INFO.keys(), help='Dataset to download')
    parser.add_argument('--overwrite', type=str2bool, default=False, help='Overwriting download dataset')
    parser.add_argument('--keep', type=str2bool, default=False, help='Keep compressed file after extracting')
    args = parser.parse_args()
    return args

def check_hash(filename, hash_type, hash_code):
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
    read_size = 1048576
    if hash_type == 'md5':
        hash_algo = hashlib.md5()
    elif hash_type == 'sha1':
        hash_algo = hashlib.sha1()
    
    with open(filename, 'rb') as f:
        while True:
            data = f.read(read_size)
            if not data:
                break
            hash_algo.update(data)

    hash_file = hash_algo.hexdigest()
    l = min(len(hash_file), len(hash_code))
    return hash_file[0:l] == hash_code[0:l]

def is_valid_url(url):
    regex = re.compile(
        r'^(?:http|ftp)s?://' # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' #domain...
        r'localhost|' #localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
        r'(?::\d+)?' # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return re.match(regex, url) is not None

def download_url(url, path=None, overwrite=False, hash_type=None, hash_code=None):
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
    hash_type : str, optional
        Type of hash algorithm
    hash_code : str, optional
        Code use for hash algoritm
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

    if overwrite or not os.path.exists(fname) or hash_type or (hash_code and not check_hash(fname, hash_type, hash_code)):
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
                for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk: # filter out keep-alive new chunks
                        f.write(chunk)
            else:
                total_length = int(total_length)
                for chunk in tqdm(r.iter_content(chunk_size=CHUNK_SIZE), total=int(total_length / CHUNK_SIZE + 0.5), unit='KB', unit_scale=False, dynamic_ncols=True):
                    f.write(chunk)

        if hash_type or (hash_code and not check_hash(fname, hash_type, hash_code)):
            raise UserWarning('File {} is downloaded but the content hash does not match. ' \
                              'The repo may be outdated or download may be incomplete. ' \
                              'If the "repo_url" is overridden, consider switching to ' \
                              'the default repo.'.format(fname))

    return fname

def extractor(file, extension, path='./', keep=False):
    """Extracting compressed file

    -------
    Args:

    file : str
        Path to compressed file
    extension : str
        Compressed file extension ('.tar', '.tar.gz', '.zip')
    path : str, optional
        Destination path to store decompressed file. By default stores to the
        current directory with same name as in url.
    keep : bool, optional
        Whether to keep compressed file after extracting
    """
    print(f"Extracting {file} ...")
    if extension in ('.tar', '.tar.gz', '.tgz'):
        with tarfile.open(file) as tar:
            members = tar.getmembers()
            extractProgress(tar, members, path)
    elif extension == '.zip':
        with zipfile.ZipFile(file, 'r') as zf:
            members = zf.infolist()
            extractProgress(zf, members, path)    
    else:
        raise "File extension must be in ('.tar', '.tar.gz', '.tgz', '.zip')"
    if not keep:
        os.remove(file)

def extractProgress(archive, members, path):
    for member in tqdm(iterable=members, total=len(members)):
        try:
            archive.extract(member=member, path=path)
        except Exception as e:       
            raise e

if __name__ == '__main__':
    args = parse_args()
    root = DATASET_INFO[args.dataset]['root']
    url = DATASET_INFO[args.dataset]['url']
    hash_attr = DATASET_INFO[args.dataset].get('hash', None)
    if hash_attr != None:
        hash_type = list(hash_attr.keys())[0]
        hash_code = hash_attr[hash_type]
    else:
        hash_type, hash_code = None, None
    ext = DATASET_INFO[args.dataset].get('extension', None)
    file = download_url(url, root, args.overwrite, hash_type, hash_code)
    extractor(file, ext, root, args.keep)