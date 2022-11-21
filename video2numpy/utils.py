"""video2numpy utils"""
import requests
import tempfile
import yt_dlp

from timeout_decorator import timeout, TimeoutError


def get_fast_format(formats, find_format_timeout):
    """returns the closest format that downloads quickly"""

    @timeout(find_format_timeout)
    def check_speed(f):
        url = f.get("url")
        ntf, _ = handle_mp4_link(url)
        with open(ntf.name, "rb") as vid_file:
            _ = vid_file.read()
        ntf.close()

    format_id = None
    for fmt in formats:
        try:
            check_speed(fmt)
            format_id = fmt.get("format_id")
            break
        except TimeoutError as _:
            pass

    return format_id


def handle_youtube(youtube_url):
    """returns file and destination name from youtube url."""
    # Probe download speed:
    ydl_opts = {
        "quiet": True,
        "external-download": "ffmpeg",
        "external-downloader-args": "ffmpeg_i:-ss 0 -t 2",  # download 2 seconds
    }
    ydl = yt_dlp.YoutubeDL(ydl_opts)
    info = ydl.extract_info(youtube_url, download=False)
    formats = info.get("formats", None)
    filtered_formats = [
        f for f in formats if f["format_note"] != "DASH video" and f["height"] is not None and f["height"] >= 360 # const 360p
    ]

    # TODO: how do we drop the video when format_id is None (all retires timed out)
    format_id = get_fast_format(filtered_formats[:10], 4)
    if format_id is None:
        return None, ""

    # Get actual video:
    # TODO: figure out a way of just requesting the format by format_id
    ydl_opts = {"quiet": True}
    ydl = yt_dlp.YoutubeDL(ydl_opts)
    info = ydl.extract_info(youtube_url, download=False)
    formats = info.get("formats", None)
    f = [f for f in formats if f["format_id"] == format_id][0]

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
        file, name = handle_youtube(url)
        return file, None, name
    elif url.endswith(".mp4"):  # mp4 link
        file, name = handle_mp4_link(url)
        return file.name, file, name
    else:
        print("Warning: Incorrect URL type")
        return None, None, ""
