#!/usr/bin/env ipython

import threading
from functools import partial
import re
import time
import os
import io
from urllib.request import urlopen
import tempfile
import logging
import tarfile

from IPython.display import display, HTML, Audio, update_display
# from ipywidgets import interact, interactive, fixed
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import soundfile

import partitura
from partitura.utils import partition
import basismixer
import basismixer.performance_codec as pc

LOGGER = logging.getLogger(__name__)

DATASET_URL = 'https://jobim.ofai.at/gitlab/accompanion/vienna4x22_rematched/repository/archive'
OGG_URL_BASE = 'https://spocs.duckdns.org/vienna_4x22/'
TMP_DIR = '/tmp/'
DATASET_DIR = None
PIECES = ()
PERFORMERS = ()
PERF_CODEC = pc.PerformanceCodec(pc.TimeCodec(), pc.NotewiseDynamicsCodec())

def init():
    global DATASET_DIR, PIECES, PERFORMERS
    status = widgets.Output()
    display(status)
    status.clear_output()

    # # assume we have the data to avoid download
    # DATASET_DIR = '/tmp/vienna4x22_rematched.git'

    if not DATASET_DIR:
        status.append_stdout('Downloading Vienna 4x22 Corpus...')
        try:
            with tarfile.open(fileobj=io.BytesIO(urlopen(DATASET_URL).read())) as archive:
                folder = next(iter(archive.getnames()), None)
                archive.extractall(TMP_DIR)
                if folder:
                    DATASET_DIR = os.path.join(TMP_DIR, folder)
        except Exception as e:
            status.append_stdout('\nError: {}'.format(e))
        status.append_stdout('done\nData is in {}'.format(DATASET_DIR))
    
    if DATASET_DIR:
        fn_pat = re.compile('(.*)_(p[0-9][0-9])\.match')
        match_files = os.listdir(os.path.join(DATASET_DIR, 'match'))
        pieces, performers = zip(*[m.groups() for m in [fn_pat.match(fn)
                                                        for fn in match_files]
                                   if m])
        PIECES = sorted(set(pieces))
        PERFORMERS = sorted(set(performers))


def load_performance_audio(piece, performer):
    url = '{}{}_{}.ogg'.format(OGG_URL_BASE, piece, performer)
    try:
        audio, fs = soundfile.read(io.BytesIO(urlopen(url).read()), always_2d=True)
        audio = audio.mean(1)
        return audio, fs
    except:
        return None, None


def get_performance_info(piece, performer, fig, axs):
    assert DATASET_DIR
    musicxml_fn = os.path.join(DATASET_DIR, 'musicxml', '{}.musicxml'.format(piece))
    match_fn = os.path.join(DATASET_DIR, 'match', '{}_{}.match'.format(piece, performer))

    part = partitura.load_musicxml(musicxml_fn)
    _, ppart, alignment = partitura.load_match(match_fn)

    part_by_id = dict((n.id, n) for n in part.notes_tied)
    ppart_by_id = dict((n['id'], n) for n in ppart.notes)

    # pair matched score and performance notes
    note_pairs = [(part_by_id[a['score_id']], #.split('-')[0]],
                   ppart_by_id[a['performance_id']])
                  for a in alignment if a['label'] == 'match']

    note_pairs.sort(key=lambda x: x[0].start.t)

    matched_score = to_matched_score(note_pairs, part.beat_map)

    targets, mbp = PERF_CODEC.encode(matched_score)
    targets[np.isnan(targets)] = 0

    bm = part.beat_map
    s_onsets = bm([n.start.t for n, _ in note_pairs])
    p_onsets = np.array([n['note_on'] for _, n in note_pairs])
    measure_times = np.array([(m.start.t, m.number) for m in
                              part.iter_all(partitura.score.Measure)])
    measure_times[:, 0] = bm(measure_times[:, 0])

    plot_targets(fig, axs, targets, onsets=s_onsets,
                 xticks=measure_times) # , title='{} {}'.format(piece, performer))

    # score_perf_map = interp1d(s_onsets, p_onsets, bounds_error=False, fill_value='extrapolate')
    score_perf_map = interp1d(s_onsets, p_onsets, bounds_error=False, fill_value=(p_onsets[0], p_onsets[-1]))

    return score_perf_map


def plot_targets(fig, axs, targets, onsets=None, xticks=None, title=None,
                 start=None, end=None):
    for ax in axs:
        ax.clear()
    
    n_targets = targets.shape[1]

    if onsets is None:
        x = np.arange(len(targets))
    else:
        x = onsets

    w = len(x)/30
    h = n_targets

    if end is not None:
        idx = x < end
        x = x[idx]
        targets = targets[idx]

    if start is not None:
        idx = x >= start
        x = x[idx]
        targets = targets[idx]
    
    if n_targets == 1:
        axs = [axs]
    
    # fig.set_size_inches(w, h)

    if title:
        fig.suptitle(title)
        
    for i, name in enumerate(PERF_CODEC.parameter_names):
        axs[i].plot(x, targets[:, i], '.', label=name)

        if xticks is not None:
            axs[i].set_xticks(xticks[:, 0])
            axs[i].set_xticklabels(xticks[:, 1])
            axs[i].xaxis.grid()

        by_onset = partition(lambda ix: ix[1], enumerate(x))
        tt = []
        vv = []
        for t, v in by_onset.items():
            tt.append(t)
            vv.append(np.mean([targets[j, i] for j, _ in v]))
        axs[i].plot(tt, vv)
        
        axs[i].legend(frameon=False, loc=2)
    return fig, axs


def performance_player():
    status = widgets.Output()
    piece_dd = widgets.Dropdown(options=PIECES, description='Piece:')
    performer_dd = widgets.Dropdown(options=PERFORMERS, description='Performer:')

    if PIECES and PERFORMERS:
        current_performance = [PIECES[0], PERFORMERS[0]]
    else:
        current_performance = [None, None]
    
    audio, fs = None, None
    score_perf_map = None
    aw = None
    fig, axs = plt.subplots(len(PERF_CODEC.parameter_names),
                            sharex=True,
                            gridspec_kw={'hspace': 0.15})
    
    def update_current_perf(info, item):
        nonlocal current_performance
        if item == 'piece':
            current_performance[0] = info['new']
        else:
            current_performance[1] = info['new']
        set_performance(*current_performance)
        
    def set_performance(piece, performer):
        nonlocal audio, fs, score_perf_map, aw
        audio, fs = load_performance_audio(piece, performer)
        score_perf_map = get_performance_info(piece, performer, fig, axs)
        if aw is None:
            aw = display(Audio(data=audio, rate=fs, autoplay=True), display_id=True)
        else:
            aw.update(Audio(data=audio, rate=fs, autoplay=True))
            
    piece_dd.observe(partial(update_current_perf, item='piece'), names=['value'])
    performer_dd.observe(partial(update_current_perf, item='performer'), names=['value'])

    display(widgets.HBox([piece_dd, performer_dd]))
    display(status)

    set_performance(*current_performance)

    cursor = []
    play_range = [None, None]
    thread_stop = None
        
    def on_mouse_down(event):
        nonlocal play_range, thread_stop
        if thread_stop:
            thread_stop.set()
        play_range[0] = event.xdata

    def on_mouse_up(event):
        nonlocal play_range, cursor, thread_stop
        play_range[1] = event.xdata
        play_range.sort()

        while cursor:
            cursor.pop().remove()

        for ax in axs:
            cursor.append(ax.fill_betweenx(ax.get_ylim(), play_range[0], play_range[1], alpha=.2, color='gray'))

        fig.canvas.draw()

        start = max(0, int(score_perf_map(play_range[0])*fs))
        end = min(len(audio), int(score_perf_map(play_range[1])*fs))
        aw.display(Audio(data=audio[start:end], rate=fs, autoplay=True))

        # duration = play_range[1] - play_range[0]
        # thread_stop = threading.Event()
        # thread = threading.Thread(
        #     target=time_cursor_thread,
        #     args=(fig, axs[0], play_range[0], play_range[1], duration, thread_stop))
        # thread.start()

    cid1 = fig.canvas.mpl_connect('button_press_event', on_mouse_down)
    cid2 = fig.canvas.mpl_connect('button_release_event', on_mouse_up)

    
def time_cursor_thread(fig, ax, start, end, duration, ev, rate=1):

    color='black'
    x = start
    vline = ax.axvline(x, c=color)
    delta_x = (end-start)/(duration*rate)
    delta_t = 1/rate

    while not ev.is_set() and x < end:

        vline.set(xdata=np.array([x, x]))
        fig.canvas.blit(ax.bbox) # doesn't reliably update
        # fig.canvas.draw()
        x += delta_x
        time.sleep(delta_t)

    vline.remove()
    fig.canvas.draw()


def to_matched_score(note_pairs, beat_map):
    ms = []
    for sn, n in note_pairs:
        sn_on, sn_off = beat_map([sn.start.t, sn.end.t])
        sn_dur = sn_off - sn_on
        n_dur = n['sound_off'] - n['note_on']
        ms.append((sn_on, sn_dur, sn.midi_pitch, n['note_on'], n_dur, n['velocity']))
    fields = [('onset', 'f4'), ('duration', 'f4'), ('pitch', 'i4'),
              ('p_onset', 'f4'), ('p_duration', 'f4'), ('velocity', 'i4')]
    return np.array(ms, dtype=fields)

init()
