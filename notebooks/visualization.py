#!/usr/bin/env ipython

import threading
from functools import partial
import time
import io
import os
import logging
from urllib.request import urlopen

from IPython.display import display, HTML, Audio, update_display
# from ipywidgets import interact, interactive, fixed
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import soundfile

import partitura
# import partitura.score as score
from partitura.utils import partition
import basismixer
import basismixer.performance_codec as pc
import data

# ensure we have the vienna4x22 corpus
data.init()

LOGGER = logging.getLogger(__name__)

OGG_URL_BASE = 'https://spocs.duckdns.org/vienna_4x22/'
# PERF_CODEC = pc.PerformanceCodec(pc.TimeCodec(), pc.NotewiseDynamicsCodec())
PERF_CODEC = pc.PerformanceCodec(pc.TimeCodec(normalization='bp_standardized'),
                                 pc.OnsetwiseDecompositionDynamicsCodec())

plt.rcParams.update({'font.size': 8})


def load_performance_audio(piece, performer):
    url = '{}{}_{}.ogg'.format(OGG_URL_BASE, piece, performer)
    try:
        audio, fs = soundfile.read(io.BytesIO(urlopen(url).read()), always_2d=True)
        audio = audio.mean(1)
        return audio, fs
    except:
        return None, None


def get_performance_info(piece, performer):
    assert data.DATASET_DIR
    musicxml_fn = os.path.join(data.DATASET_DIR, 'musicxml', '{}.musicxml'.format(piece))
    match_fn = os.path.join(data.DATASET_DIR, 'match', '{}_{}.match'.format(piece, performer))

    part = partitura.load_musicxml(musicxml_fn)
    
    ppart, alignment = partitura.load_match(match_fn, first_note_at_zero=True)
    return part, ppart, alignment


def show_performance(piece, performer, fig, axs, keep_zoom):
    part, ppart, alignment = get_performance_info(piece, performer)
    targets, snote_ids = PERF_CODEC.encode(part, ppart, alignment)

    # we convert to f8 to avoid numerical problems when computing means
    dtype = [(n, 'f8') for n in targets.dtype.names]
    targets = targets.astype(dtype)
        
    part_by_id = dict((n.id, n) for n in part.notes_tied)
    ppart_by_id = dict((n['id'], n) for n in ppart.notes)
    s_to_p_id = dict((a['score_id'], a['performance_id'])
                     for a in alignment if a['label'] == 'match')
    s_notes = [part_by_id[n] for n in snote_ids]
    p_notes = [ppart_by_id[s_to_p_id[n]] for n in snote_ids]

    bm = part.beat_map
    s_onsets = bm([n.start.t for n in s_notes])
    p_onsets = np.array([n['note_on'] for n in p_notes])
    measure_times = np.array([(m.start.t, '{}'.format(m.number)) for m in
                              part.iter_all(partitura.score.Measure)],
                             dtype=[('t', 'f4'), ('label', 'U100')])

    measure_times['t'] = bm(measure_times['t'])

    make_plot(fig, axs, targets, onsets=s_onsets, xlabel='Measure number',
              xticks=measure_times, keep_zoom=keep_zoom) # , title='{} {}'.format(piece, performer))

    s_times = np.r_[s_onsets, s_notes[-1].end.t]
    p_times = np.r_[p_onsets, p_notes[-1]['note_off']]
    # score_perf_map = interp1d(s_onsets, p_onsets, bounds_error=False, fill_value='extrapolate')
    score_perf_map = interp1d(s_times, p_times, bounds_error=False, fill_value=(p_times[0], p_times[-1]))

    return score_perf_map


def make_plot(fig, axs, targets, onsets=None, xticks=None, title=None,
                 xlabel=None, start=None, end=None, keep_zoom=False):
    names = targets.dtype.names

    xlims = []
    ylims = []
    for ax in axs:
        if keep_zoom:
            xlims.append(list(ax.get_xlim()))
            ylims.append(list(ax.get_ylim()))
        ax.clear()

    n_targets = len(names)

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

    by_onset = partition(lambda ix: ix[1], enumerate(x))
    for k, v in by_onset.items():
        by_onset[k] = np.array([i for i, _ in v])

    for i, name in enumerate(names):
        target = targets[name]
        targets[np.isnan(target)] = 0
        
        axs[i].plot(x, target, '.', label=name)

        if xticks is not None:
            axs[i].set_xticks(xticks['t'])
            axs[i].set_xticklabels(xticks['label'])
            axs[i].xaxis.grid()

        tt = []
        vv = []
        for t, v in by_onset.items():
            tt.append(t)
            vv.append(np.mean(target[v]))

        # axs[i].plot(tt, vv, label='{} (mean)'.format(name))
        axs[i].plot(tt, vv)
        
        axs[i].legend(frameon=False, loc=2)

    if keep_zoom:
        axs[0].set_xlim(xlims[0])
        for xlim, ylim, ax in zip(xlims, ylims, axs):
            ax.set_ylim(ylim)

    return fig, axs


def performance_player():
    status = widgets.Output()
    piece_dd = widgets.Dropdown(options=data.PIECES, description='Piece:')
    performer_dd = widgets.Dropdown(options=data.PERFORMERS, description='Performer:')
    keep_lims_chbox = widgets.Checkbox(value=False, description='Keep zoom')
    reset_lims = widgets.Button(description='Zoom to fit',
                                button_style='', # 'success', 'info', 'warning', 'danger' or ''
                                tooltip='Zoom to fit',
                                icon='check'
                                )


    if data.PIECES and data.PERFORMERS:
        current_performance = [data.PIECES[0], data.PERFORMERS[0]]
    else:
        current_performance = [None, None]
    
    audio, fs = None, None
    score_perf_map = None
    aw = None
    keep_zoom = False

    fig, axs = plt.subplots(len(PERF_CODEC.parameter_names),
                            sharex=True,
                            gridspec_kw={'hspace': 0.15})
    plt.subplots_adjust(left=0.07, right=0.99, top=.99, bottom=0.1)
    
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
        score_perf_map = show_performance(piece, performer, fig, axs, keep_zoom)

        if keep_zoom:
            s, e = axs[0].get_xlim()
            start = max(0, int(score_perf_map(s)*fs))
            end = min(len(audio), int(score_perf_map(e)*fs))
            excerpt = audio[start:end]
        else:
            excerpt = audio
        if aw is None:
            aw = display(Audio(data=excerpt, rate=fs, autoplay=True), display_id=True)
        else:
            aw.update(Audio(data=excerpt, rate=fs, autoplay=True))

    def set_keep_zoom(v):
        nonlocal keep_zoom
        keep_zoom = v['new']
    
    def do_reset_zoom(v):
        nonlocal axs, fig
        for ax in axs:
            ax.autoscale()
            ax.autoscale_view()
        fig.canvas.draw()  

    piece_dd.observe(partial(update_current_perf, item='piece'), names=['value'])
    performer_dd.observe(partial(update_current_perf, item='performer'), names=['value'])
    keep_lims_chbox.observe(set_keep_zoom, names=['value'])
    reset_lims.on_click(do_reset_zoom)

    display(widgets.HBox([piece_dd, performer_dd, keep_lims_chbox, reset_lims]))
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
        fig.canvas.draw()
        vline.set(xdata=np.array([x, x]))
        # fig.canvas.blit(ax.bbox) # doesn't reliably update
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
