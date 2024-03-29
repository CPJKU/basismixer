from partitura.utils import ensure_notearray
import numpy as np
import os


def alignment_dicts_to_array(alignment):
    """
    create structured array from list of dicts type alignment.

    Parameters
    ----------
    alignment : list
        A list of note alignment dictionaries.

    Returns
    -------
    alignarray : structured ndarray
        Structured array containing note alignment.
    """
    fields = [('idx', 'i4'),
              ('matchtype', 'U256'),
              ('partid', 'U256'),
              ('ppartid', 'U256')]

    array = []
    # for all dicts create an appropriate entry in an array:
    # match = 0, deletion  = 1, insertion = 2
    for no, i in enumerate(alignment):
        if i["label"] == "match":
            array.append((no, "0", i["score_id"], str(i["performance_id"])))
        elif i["label"] == "insertion":
            array.append((no, "2", "undefined", str(i["performance_id"])))
        elif i["label"] == "deletion":
            array.append((no, "1", i["score_id"], "undefined"))
    alignarray = np.array(array, dtype=fields)

    return alignarray


def save_csv_for_parangonada(outdir, part, ppart, align,
                             zalign=None, feature=None):
    """
    Save an alignment for visualization with parangonda.

    Parameters
    ----------
    outdir : str
        A directory to save the files into.
    part : Part, structured ndarray
        A score part or its note_array.
    ppart : PerformedPart, structured ndarray
        A PerformedPart or its note_array.
    align : list
        A list of note alignment dictionaries.
    zalign : list, optional
        A second list of note alignment dictionaries.
    feature : list, optional
        A list of expressive feature dictionaries.

    """

    part = ensure_notearray(part)
    ppart = ensure_notearray(ppart)

    ffields = [('velocity', '<f4'),
               ('timing', '<f4'),
               ('articulation', '<f4'),
               ('id', 'U256')]

    farray = []
    notes = list(part["id"])
    if feature is not None:
        # veloctiy, timing, articulation, note
        for no, i in enumerate(list(feature['id'])):
            farray.append((feature['velocity'][no], feature['timing'][no],
                           feature['articulation'][no], i))
    else:
        for no, i in enumerate(notes):
            farray.append((0, 0, 0, i))

    featurearray = np.array(farray, dtype=ffields)
    alignarray = alignment_dicts_to_array(align)

    if zalign is not None:
        zalignarray = alignment_dicts_to_array(zalign)
    else:  # if no zalign is available, save the same alignment twice
        zalignarray = alignment_dicts_to_array(align)

    np.savetxt(outdir + os.path.sep + "Nppart.csv", ppart,
               fmt="%.20s", delimiter=",", header=",".join(ppart.dtype.names), comments="")
    np.savetxt(outdir + os.path.sep + "Npart.csv", part,
               fmt="%.20s", delimiter=",", header=",".join(part.dtype.names), comments="")
    np.savetxt(outdir + os.path.sep + "Nalign.csv", alignarray,
               fmt="%.20s", delimiter=",", header=",".join(alignarray.dtype.names), comments="")
    np.savetxt(outdir + os.path.sep + "Nzalign.csv", zalignarray,
               fmt="%.20s", delimiter=",", header=",".join(zalignarray.dtype.names), comments="")
    np.savetxt(outdir + os.path.sep + "Nfeature.csv", featurearray,
               fmt="%.20s", delimiter=",", header=",".join(featurearray.dtype.names), comments="")


def load_alignment_from_parangonada(outfile):
    """
    load an alignment exported from parangonda.

    Parameters
    ----------
    outfile : str
        A path to the alignment csv file

    Returns
    -------
    alignlist : list
        A list of note alignment dictionaries.
    """
    array = np.loadtxt(outfile, dtype=str, delimiter=",")
    alignlist = list()
    # match = 0, deletion  = 1, insertion = 2
    for k in range(1, array.shape[0]):
        if array[k, 1] == 0:
            alignlist.append({"label": "match", "score_id": array[k, 2], "performance_id": array[k, 3]})

        elif array[k, 1] == 2:
            alignlist.append({"label": "insertion", "performance_id": array[k, 3]})

        elif array[k, 1] == 0:
            alignlist.append({"label": "deletion", "score_id": array[k, 2]})
    return alignlist


def save_tsv_for_ASAP(outfile, ppart, alignment):
    """
    load an alignment exported from parangonda.

    Parameters
    ----------
    outfile : str
        A path for the alignment tsv file.
    ppart : PerformedPart, structured ndarray
        A PerformedPart or its note_array.
    align : list
        A list of note alignment dictionaries.

    """
    notes_indexed_by_id = {str(n["id"]): [str(n["id"]),
                                          str(n["track"]),
                                          str(n["channel"]),
                                          str(n["midi_pitch"]),
                                          str(n["note_on"])]
                           for n in ppart.notes}
    with open(outfile, 'w') as f:
        f.write('xml_id\tmidi_id\ttrack\tchannel\tpitch\tonset\n')
        for line in alignment:
            if line["label"] == "match":
                outline_score = [str(line["score_id"])]
                outline_perf = notes_indexed_by_id[str(line["performance_id"])]
                f.write('\t'.join(outline_score + outline_perf) + '\n')
            elif line["label"] == "deletion":
                outline_score = str(line["score_id"])
                f.write(outline_score + '\tdeletion\n')
            elif line["label"] == "insertion":
                outline_score = ["insertion"]
                outline_perf = notes_indexed_by_id[str(line["performance_id"])]
                f.write('\t'.join(outline_score + outline_perf) + '\n')


def load_alignment_from_ASAP(file):
    """
    load a note alignment of the ASAP dataset.

    Parameters
    ----------
    file : str
        A path to the alignment tsv file

    Returns
    -------
    alignlist : list
        A list of note alignment dictionaries.
    """
    alignlist = list()
    with open(file, 'r') as f:
        for line in f.readlines()[1:]:  # skip header
            fields = line.split("\t")
            if fields[0][0] == "n" and not fields[1].startswith("deletion"):
                field0 = fields[0]#.split("-")[0] # todo: how to handle 'n123-x' when x > 1, all quirk?
                alignlist.append({"label": "match", "score_id": field0, "performance_id": fields[1]})
            elif fields[0] == "insertion":
                alignlist.append({"label": "insertion", "performance_id": fields[1]})
            elif fields[0][0] == "n" and fields[1].startswith("deletion"):
                field0 = fields[0]#.split("-")[0]
                alignlist.append({"label": "deletion", "score_id": field0})
            else:
                raise Exception(f"Unknown alignment type: {fields[0]}")

    return alignlist