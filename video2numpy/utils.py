"""video2numpy utils"""


def split_block(block, ind_dict):
    """separate block into individual videos using ind_dict"""
    sep_frames = {}
    for dst_name, inds in ind_dict.items():
        i0, it = inds
        vid_frames = block[i0:it]

        sep_frames[dst_name] = vid_frames
    return sep_frames
