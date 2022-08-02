"""video2numpy utils"""
import youtube_dl


QUALITY = "360p"


def handle_youtube(url, modality):
    """returns file and destination name from youtube url."""
    cv2_vid = None
    info = None

    # for video, ydl_opts = {}
    if modality == "video":
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

    # for audio, ydl_opts = {"format": "bestaudio"}, aiming at extracting only audio
    elif modality == "audio":
        ydl_opts = {
            "format": "bestaudio"
        }
        ydl = youtube_dl.YoutubeDL(ydl_opts)
        info = ydl.extract_info(url, download=False)
        cv2_vid = info.get("url", None)

    # Common for both video and audio: youtube id
    dst_name = info.get("id") + ".npy"

    return cv2_vid, dst_name

