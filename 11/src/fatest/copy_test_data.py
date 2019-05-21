import os
import argparse
from shutil import copy
from functools import partial
from multiprocessing import Pool

from tqdm import tqdm


def copy_file(file, source_dir, target_dir):
    source_path = os.path.join(source_dir, file)
    target_path = os.path.join(target_dir, file)
    if not os.path.exists(os.path.dirname(target_path)):
        try: os.makedirs(os.path.dirname(target_path))
        except: pass
    copy(source_path, target_path)


def main(args):
    content = [l.strip().split()[:-1] for l in open(args.list_path)]
    all_files = set()
    for l in content:
        for f in l:
            all_files.add(f)
    func = partial(copy_file, source_dir=args.source_dir, target_dir=args.target_dir)
    with Pool(10) as p:
        for _ in tqdm(p.imap(func, all_files), total=len(all_files)):
            pass


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--list_path', required=True)
    p.add_argument('--source_dir', required=True)
    p.add_argument('--target_dir', required=True)
    main(p.parse_args())
