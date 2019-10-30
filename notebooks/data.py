#!/usr/bin/env python

import os
import argparse
import tarfile
import io
from urllib.request import urlopen
import re

from IPython.display import display, HTML, Audio, update_display
import ipywidgets as widgets
import appdirs

from basismixer.utils import pair_files

REPO_NAME = 'vienna4x22_rematched'
DATASET_URL = 'https://jobim.ofai.at/gitlab/accompanion/{}/repository/archive'.format(REPO_NAME)
OGG_URL_BASE = 'https://spocs.duckdns.org/vienna_4x22/'
TMP_DIR = appdirs.user_cache_dir('basismixer')
# this is where our data set will be
DATASET_DIR = os.path.join(TMP_DIR, '{}.git'.format(REPO_NAME))
PIECES = ()
PERFORMERS = ()
SCORE_PERFORMANCE_PAIRS = None

def init():
    global DATASET_DIR, PIECES, PERFORMERS, SCORE_PERFORMANCE_PAIRS

    status = widgets.Output()
    display(status)
    status.clear_output()

    # # assume we have the data to avoid download
    # DATASET_DIR = '/tmp/vienna4x22_rematched.git'
    
    if os.path.exists(DATASET_DIR):
        status.append_stdout('Vienna 4x22 Corpus already downloaded.\n')
        status.append_stdout('Data is in {}'.format(DATASET_DIR))
    else:
        status.append_stdout('Downloading Vienna 4x22 Corpus...')
        try:
            with tarfile.open(fileobj=io.BytesIO(urlopen(DATASET_URL).read())) as archive:
                folder = next(iter(archive.getnames()), None)
                archive.extractall(TMP_DIR)
                # if folder:
                #     DATASET_DIR = os.path.join(TMP_DIR, folder)
                assert DATASET_DIR == os.path.join(TMP_DIR, folder)
                
        except Exception as e:
            status.append_stdout('\nError: {}'.format(e))
            return
        status.append_stdout('done\nData is in {}'.format(DATASET_DIR))
    
    folders = dict(musicxml=os.path.join(DATASET_DIR, 'musicxml'),
                   match=os.path.join(DATASET_DIR, 'match'))

    SCORE_PERFORMANCE_PAIRS = []
    paired_files = pair_files(folders)
    pieces = sorted(paired_files.keys())
    for piece in pieces:
        xml_fn = paired_files[piece]['musicxml'].pop()
        for match_fn in sorted(paired_files[piece]['match']):
            SCORE_PERFORMANCE_PAIRS.append((xml_fn, match_fn))
            
    fn_pat = re.compile('(.*)_(p[0-9][0-9])\.match')
    match_files = os.listdir(os.path.join(DATASET_DIR, 'match'))
    pieces, performers = zip(*[m.groups() for m in [fn_pat.match(fn)
                                                    for fn in match_files]
                               if m])
    PIECES = sorted(set(pieces))
    PERFORMERS = sorted(set(performers))
