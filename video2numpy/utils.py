"""video2numpy utils"""
import requests
import tempfile
import yt_dlp
# from yt_dlp.utils import download_range_func

QUALITY = "360p"


def handle_youtube(youtube_url):
    """returns file and destination name from youtube url."""
    '''
    ydl_opts = {
        "download_ranges": download_range_func(None, [(0, 5)])
    }
    '''
    # ydl_opts = {}
    ydl_opts = {"format": "bestvideo*+bestaudio/best"}

    ydl = yt_dlp.YoutubeDL(ydl_opts)
    info = ydl.extract_info(youtube_url, download=False, force_generic_extractor=True)
    formats = info.get("formats", None)

    f = None
    for f in formats:
        # if f.get("format_note", None) != QUALITY:
        if f.get("vcodec", None) == 'none':
            continue
        break
    print(f)

    cv2_vid = f.get("url", None)
    dst_name = info.get("id") + ".npy"
    return cv2_vid, dst_name


def handle_mp4_link(mp4_link):
    resp = requests.get(mp4_link, stream=True)
    ntf = tempfile.NamedTemporaryFile()  # pylint: disable=consider-using-with
    ntf.write(resp.content)
    ntf.seek(0)
    dst_name = mp4_link.split("/")[-1][:-4] + ".npy"
    return ntf, dst_name


def handle_url(url):
    """
    Input:
        url: url of video

    Output:
        load_file - variable used to load video.
        file - the file itself (in cases where it needs to be closed after usage).
        name - numpy fname to save frames to.
    """
    if "youtube" in url:  # youtube link
        load_file, name = handle_youtube(url)
        return load_file, None, name
    elif url.endswith(".mp4"):  # mp4 link
        file, name = handle_mp4_link(url)
        return file.name, file, name
    else:
        print("Warning: Incorrect URL type")
        return None, None, ""
