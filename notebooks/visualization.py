#!/usr/bin/env ipython

import threading
from IPython.display import display, HTML, Audio, update_display
import time
# from ipywidgets import interact, interactive, fixed
# import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import soundfile

def play_audio_fragment(audio_fn):
    x = np.random.random(10)
    # out = widgets.Output()
    audio, fs = soundfile.read(audio_fn)
    audio = audio.mean(1)
    # audio_widget = Audio(filename=fragment_fn, autoplay=True)
    # aw = display(audio_widget, display_id=True)
    aw = display(Audio(url='', embed=False), display_id=True)
    fig, ax = plt.subplots()
    ax.plot(np.random.rand(100))
    ax.set_ylabel('Target')
    ax.set_xlabel('Time (seconds)')
    cursor = None
    play_range = [None, None]
    thread_stop = None

    def on_mouse_down(event):
        nonlocal play_range, thread_stop
        if thread_stop:
            thread_stop.set()
        play_range[0] = event.xdata

    def on_mouse_up(event):
        nonlocal play_range, cursor, thread_stop
        # out.append_stdout(f'{play_range}')
        play_range[1] = event.xdata
        play_range.sort()
        if cursor:
            cursor.remove()
        cursor = ax.fill_betweenx([0, 1], play_range[0], play_range[1], alpha=.2, color='gray')
        fig.canvas.draw()
        start = int(play_range[0]*fs)
        end = int(play_range[1]*fs)
        aw.display(Audio(data=audio[start:end], rate=fs, autoplay=True))
        duration = play_range[1] - play_range[0]
        thread_stop = threading.Event()
        thread = threading.Thread(
            target=time_cursor_thread,
            args=(fig, ax, play_range[0], play_range[1], duration, thread_stop))
        thread.start()

    # cid = fig.canvas.mpl_connect('button_press_event', onclick)
    cid1 = fig.canvas.mpl_connect('button_press_event', on_mouse_down)
    cid2 = fig.canvas.mpl_connect('button_release_event', on_mouse_up)
    # fig.canvas.draw()
    # display(out)
    
def time_cursor_thread(fig, ax, start, end, duration, ev, rate=1):

    color='black'
    x = start
    vline = ax.axvline(x, c=color)
    delta_x = (end-start)/(duration*rate)
    delta_t = 1/rate

    while not ev.is_set() and x < end:

        vline.set(xdata=np.array([x, x]))
        # fig.canvas.blit(ax.bbox) # doesn't reliably update
        fig.canvas.draw()
        x += delta_x
        time.sleep(delta_t)

    vline.remove()
    fig.canvas.draw()
    
# def thread_func(audio, out):
#     for i in range(1, 5):
#         time.sleep(0.5)
#         out.append_stdout('{} {}\n'.format(i, '**'*i))
#     out.append_stdout('{}\n'.format(dir(audio)))
#     out.append_display_data(HTML("<em>All done!</em>"))

# def main():
#     # display('Display in main thread')
#     audio_fn = 'test_audio.flac'
#     audio = Audio(filename=audio_fn)
#     out = widgets.Output()
#     # Now the key: the container is displayed (while empty) in the main thread
#     thread = threading.Thread(
#         target=thread_func,
#         args=(audio, out))
#     display(audio)
#     thread.start()
#     display(out)


        
    # def onclick(event):
    #     nonlocal fragment, audio_widget, cursor
    #     # out.append_stdout('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
    #     #       ('double' if event.dblclick else 'single', event.button,
    #     #        event.x, event.y, event.xdata, event.ydata))
    #     start = int(event.xdata*fs)
    #     end = start + N
    #     # soundfile.write(fragment_fn, audio[start:end], fs)
    #     # audio_widget.reload()
    #     if cursor:
    #         cursor.remove()
    #     cursor = ax.axvline(event.xdata)
    #     fig.canvas.draw()
    #     aw.display(Audio(data=audio[start:end], rate=fs, autoplay=True))
    #     out.append_stdout(f'{audio.shape} {int(start*fs)} {int(end*fs)}\n')
    #     # fragment[:] = audio[start:end]
