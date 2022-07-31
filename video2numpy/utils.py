"""video2numpy utils"""
import youtube_dl
import librosa  
import os


QUALITY = "360p"
SAMPLING_RATE = 48000


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

# Unfortunately, I did not find a way to convert directly from the url
# to a numpy ndarray. So I have to download the audio into the disk and
# reload it using librosa.

# In fact such way exists, using numpy.frombuffer(requests.get(url).content))
# However if I do so I can not ensure that the audio 
# format is wav, neither can I reset the sampling rate to 48000.
 
 
def extract_audio_from_url(url, output_dir):
    #returns a numpy array and destination name of the audio extracted

    # extract the audio from the url
    ydl_opts = {
        "outtmpl": output_dir +"/"+"%(id)s.%(ext)s", 
        # since ffmpeg can not edit file in place, input and output file name can not be the same
        "format": "bestaudio/best",
        "postprocessors": [
            {   
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav"
            },
        ]
    }

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
        info = ydl.extract_info(url, download=False)
        id = info["id"]
        file_path = output_dir + "/" + id + ".wav"

        # librosa.org/doc/latest/generated/librosa.load.html#librosa.load
        # Convert to numpy array
        array, _ = librosa.load(file_path, sr=SAMPLING_RATE, mono = True)

        # remove the transistory .wav file
        os.remove(file_path)
        return array , id + ".npy"
