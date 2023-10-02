import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst

def main():
    # Inicializar la biblioteca de GStreamer

    Gst.init(None)

    # Crear el pipeline

    pipeline_video = Gst.Pipeline()

    # Crear el elemento filesrc

    filesrc = Gst.ElementFactory.make("filesrc")
    filesrc.set_property("location", "coco.mp4")

    # Agregar el elemento filesrc al pipeline

    pipeline_video.add(filesrc)

    # Crear el elemento windowsink

    windowsink = Gst.ElementFactory.make("windowsink")

    # Agregar el elemento windowsink al pipeline

    pipeline_video.add(windowsink)

    # Establecer la propiedad x

    windowsink.set_property("x", 100)
    windowsink.set_property("y", 100)
    windowsink.set_property("width", 640)
    windowsink.set_property("height", 480)

    # Iniciar el pipeline

    pipeline_video.set_state(Gst.State.PLAYING)

    # Esperar a que el pipeline termine

    pipeline_video.get_state(Gst.ClockTime.TIME_NONE)

if __name__ == "__main__":
    main()
