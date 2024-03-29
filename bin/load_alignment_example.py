#!/usr/bin/env python

import argparse

import matplotlib.pyplot as plt
import numpy as np
import partitura
from partitura.utils import partition
import basismixer.performance_codec as pc
from basismixer.utils import pair_files


def main():
    parser = argparse.ArgumentParser(
        description="Extract match information for performance codec"
    )
    parser.add_argument("xmlfolder", help="Folder with MusicXML files")
    parser.add_argument("matchfolder", help="Folder with match files")
    parser.add_argument("--valid-pieces", help="Valid pieces", default=None)
    args = parser.parse_args()

    folders = dict(xml=args.xmlfolder, match=args.matchfolder)

    valid_pieces = None
    if args.valid_pieces:
        valid_pieces = np.loadtxt(args.valid_pieces, dtype=str)

    # for piece, files in pair_files(folders).items():
    for piece, files in pair_files(folders, by_prefix=False).items():
        if valid_pieces is not None:
            if piece not in valid_pieces:
                continue
        print(piece)

        part = partitura.load_musicxml(files["xml"].pop(), validate=False)
        # for n in part.iter_all(partitura.score.GenericNote, include_subclasses=True):
        #     if not isinstance(n, partitura.score.GraceNote):
        #         n.symbolic_duration = None
        partitura.score.expand_grace_notes(part)

        # with open('/tmp/pretty.txt', 'w') as pretty:
        #     pretty.write(part.pretty())

        try:
            for match in files["match"]:
                ppart, alignment = partitura.load_match(match, first_note_at_zero=True)

                for a in alignment:
                    if "score_id" in a:
                        a["score_id"] = a["score_id"].split("-")[0]

                dyn_codec = pc.OnsetwiseDecompositionDynamicsCodec()
                time_codec = pc.TimeCodec(
                    normalization="beat_period_standardized",
                )
                perf_codec = pc.PerformanceCodec(time_codec, dyn_codec)
                targets, snote_ids = perf_codec.encode(part, ppart, alignment)

                rec_perf = perf_codec.decode(part, targets, snote_ids=snote_ids)

                # compare reconstructed performance to original performance
                id_map = dict(
                    (a["score_id"], a["performance_id"])
                    for a in alignment
                    if a["label"] == "match"
                )
                note_ids = sorted(id_map.keys())
                orig_note_dict = dict((n["id"], n) for n in ppart.notes)
                orig_notes = [orig_note_dict[id_map[i]] for i in snote_ids]
                rec_note_dict = dict((n["id"], n) for n in rec_perf.notes)
                rec_notes = [rec_note_dict[i] for i in snote_ids]

                rtol = 1e-5
                atol = 1e-5
                velocities = np.array(
                    [
                        (n["velocity"], rn["velocity"])
                        for n, rn in zip(orig_notes, rec_notes)
                    ]
                )
                if not all(
                    np.isclose(velocities[:, 0], velocities[:, 1], rtol=rtol, atol=atol)
                ):
                    plot_reconstructed_targets("velocities", velocities)

                onsets = np.array(
                    [
                        (n["note_on"], rn["note_on"])
                        for n, rn in zip(orig_notes, rec_notes)
                    ]
                )
                # original matched performance notes may not start at 0 under all
                # circumstances (e.g. when the first performance note is not matched
                # to a score note). Therefore we shift the times by delta to make
                # sure the original matched performance notes start at zero.
                delta = np.min(onsets[:, 0])
                onsets[:, 0] -= delta

                if not np.all(
                    np.isclose(onsets[:, 0], onsets[:, 1], rtol=rtol, atol=atol)
                ):
                    plot_reconstructed_targets("onsets", onsets)

                offsets = np.array(
                    [
                        (n["sound_off"], rn["sound_off"])
                        for n, rn in zip(orig_notes, rec_notes)
                    ]
                )
                offsets[:, 0] -= delta

                # if not np.all(np.isclose(offsets[:, 0], offsets[:, 1],
                #                          rtol=rtol, atol=atol)):
                #     plot_reconstructed_targets('offsets', offsets)

                # plot targets
                bm = part.beat_map
                snote_dict = dict((n.id, n) for n in part.notes_tied)
                # onsets = bm([n.start.t for n, _ in note_pairs])
                onsets = bm([snote_dict[i].start.t for i in snote_ids])
                measure_times = np.array(
                    [
                        (m.start.t, m.number)
                        for m in part.iter_all(partitura.score.Measure)
                    ]
                )
                measure_times[:, 0] = bm(measure_times[:, 0])

                plot_targets(
                    targets,
                    perf_codec.parameter_names,
                    "/tmp/{0}.png".format(piece),
                    onsets=onsets,
                    xticks=measure_times,
                    title=match,
                )

        except:
            print("error in piece:", piece)


def plot_reconstructed_targets(title, values):
    plt.clf()
    plt.title(title)
    plt.plot(values[:, 0], label="orig")
    plt.plot(values[:, 1], label="rec")
    plt.grid()
    plt.legend()
    plt.show()


def plot_targets(targets, names, out_fn, onsets=None, xticks=None, title=None):

    n_targets = len(targets.dtype.names)

    if onsets is None:
        x = np.arange(len(targets))
    else:
        x = onsets

    w = len(x) / 30
    h = n_targets

    fig, axs = plt.subplots(n_targets, sharex=True, gridspec_kw={"hspace": 0.15})
    if n_targets == 1:
        axs = [axs]

    fig.set_size_inches(w, h)

    if title:
        fig.suptitle(title)

    for i, name in enumerate(names):
        axs[i].plot(x, targets[name], ".", label=name)

        if xticks is not None:
            axs[i].set_xticks(xticks[:, 0])
            axs[i].set_xticklabels(xticks[:, 1])
            axs[i].xaxis.grid()

        by_onset = partition(lambda ix: ix[1], enumerate(x))
        tt = []
        vv = []
        for t, v in by_onset.items():
            tt.append(t)
            vv.append(np.mean([targets[name][j] for j, _ in v]))

        # ymax = np.mean(vv) + 2 * np.std(vv)
        # ymin = np.mean(vv) - 2 * np.std(vv)
        ymax = np.mean(vv) + 3 * np.std(targets[name])
        ymin = np.mean(vv) - 3 * np.std(targets[name])

        axs[i].set_ylim(
            (np.minimum(ymin, np.mean(vv) - 0.1), np.maximum(ymax, np.mean(vv) + 0.1))
        )

        # ymin = np.min(np.r_[vv, targets[name]])
        # ymax = np.max(np.r_[vv, targets[name]])

        # axs[i].set_ylim((ymin, ymax))

        axs[i].plot(tt, vv)

        axs[i].legend(frameon=False, loc=1)

    fig.savefig(out_fn, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
