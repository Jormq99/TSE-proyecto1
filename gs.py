import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst, Gtk

Gst.init(None)

class MyApp(Gtk.Window):

    def __init__(self):
        Gtk.Window.__init__(self, title="Reproducción de Video")

        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.add(vbox)

        # Creamos los elementos de GStreamer
        self.pipeline = Gst.Pipeline()
        self.filesrc = Gst.ElementFactory.make('filesrc', 'file-source')
        self.decodebin = Gst.ElementFactory.make('decodebin', 'decode-bin')
        self.autovideosink = Gst.ElementFactory.make('autovideosink', 'auto-video-sink')

        # Agregamos los elementos al pipeline
        self.pipeline.add(self.filesrc)
        self.pipeline.add(self.decodebin)
        self.pipeline.add(self.autovideosink)

        # Conectamos los elementos
        self.filesrc.link(self.decodebin)
        self.decodebin.link(self.autovideosink)

        # Configuramos el archivo de origen
        self.filesrc.set_property('location', 'coco.mp4')

        # Iniciamos el pipeline
        self.pipeline.set_state(Gst.State.PLAYING)

win = MyApp()
win.connect("destroy", Gtk.main_quit)
win.show_all()
Gtk.main()
