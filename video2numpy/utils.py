"""video2numpy utils"""
import requests
import tempfile
import youtube_dl


QUALITY = "360p"


def handle_youtube(youtube_url):
    """returns file and destination name from youtube url."""
    ydl_opts = {}
    ydl = youtube_dl.YoutubeDL(ydl_opts)
    info = ydl.extract_info(youtube_url, download=False)
    formats = info.get("formats", None)
    f = None
    for f in formats:
        if f.get("format_note", None) != QUALITY:
            continue
        break

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
