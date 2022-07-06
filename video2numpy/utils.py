"""video2numpy utils"""


def split_block(block, ind_dict):
    """
    separate block into individual videos using ind_dict

    Input:
      block - numpy array returned from FrameReader or some derivative of it
              with the same batch dimension order
      ind_dict - dict that shows what indices correspond to what rows of the "block" arg
                 {"video_name": (block_ind0, block_indf) ...}
    Output:
      dict - {"video_name": stacked block rows corresponding to that video (usually frames) ...}
    """
    sep_frames = {}
    for dst_name, inds in ind_dict.items():
        i0, it = inds
        vid_frames = block[i0:it]

        sep_frames[dst_name] = vid_frames
    return sep_frames
