"""video2numpy utils"""
import requests
import tempfile
import yt_dlp


QUALITY = "360p"

# TODO make this better / audio support
def get_format_selector(retry):
    """
    Gets format selector based on retry number.
    """
    def format_selector(ctx):
        formats = ctx.get('formats')
        if retry == 0:
            for f in formats:
                if f.get("format_note", None) != QUALITY:
                    continue
                break
        else:
            for f in formats: # take WORST video format available
                if f.get("vcodec", None) == 'none':
                    continue
                break
        yield {
            "format_id": f["format_id"],
            "ext": f["ext"],
            "requested_formats": [f],
            "protocol": f["protocol"],
        }
    return format_selector


def handle_youtube(youtube_url, retry):
    """returns file and destination name from youtube url."""

    ydl_opts = {
        "quiet": True,
        "format": get_format_selector(retry),
    }

    ydl = yt_dlp.YoutubeDL(ydl_opts)
    info = ydl.extract_info(youtube_url, download=False)
    formats = info.get("requested_formats", None)
    f = formats[0]

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


def handle_url(url, retry):
    """
    Input:
        url: url of video

    Output:
        load_file - variable used to load video.
        file - the file itself (in cases where it needs to be closed after usage).
        name - numpy fname to save frames to.
    """
    if "youtube" in url:  # youtube link
        load_file, name = handle_youtube(url, retry)
        return load_file, None, name
    elif url.endswith(".mp4"):  # mp4 link
        file, name = handle_mp4_link(url)
        return file.name, file, name
    else:
        print("Warning: Incorrect URL type")
        return None, None, ""
