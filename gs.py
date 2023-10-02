from time import sleep
from threading import Thread

import gi

gi.require_version("Gst", "1.0")

from gi.repository import Gst, GLib


Gst.init()

main_loop = GLib.MainLoop()
thread = Thread(target=main_loop.run)
thread.start()

pipeline_video = Gst.parse_launch("udpsrc multicast-group=224.1.1.1 auto-multicast=true port=5001 ! application/x-rtp  ! rtph264depay ! h264parse ! queue ! decodebin ! videoconvert ! autovideosink")
pipeline_audio = Gst.parse_launch("udpsrc multicast-group=224.1.1.1 auto-multicast=true port=5000 ! rawaudioparse use-sink-caps=false format=pcm pcm-format=s16le sample-rate=16000 num-channels=1 ! queue ! audioconvert ! audioresample ! autoaudiosink")
pipeline_video.set_state(Gst.State.PLAYING)

pipeline_audio.set_state(Gst.State.PLAYING)

try:
    while True:
        sleep(0.1)
except KeyboardInterrupt:
    pass
pipeline_video.set_state(Gst.State.NULL)
pipeline_audio.set_state(Gst.State.NULL)
main_loop.quit()
