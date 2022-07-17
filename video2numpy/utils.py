"""video2numpy utils"""
import youtube_dl


QUALITY = "360p"


def handle_youtube(url):
    """returns file and destination name from youtube url."""
    ydl_opts = {}
    ydl = youtube_dl.YoutubeDL(ydl_opts)
    info = ydl.extract_info(url, download=False)
    formats = info.get("formats", None)
    f = None
    for f in formats:
        if f.get("format_note", None) != QUALITY:
            continue
        break

    cv2_vid = f.get("url", None)
    dst_name = info.get("id") + ".npy"
    return cv2_vid, dst_name
