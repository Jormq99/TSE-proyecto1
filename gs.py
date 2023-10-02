import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst, Gtk

Gst.init(None)

class MyApp(Gtk.Window):

    def __init__(self):
        Gtk.Window.__init__(self, title="Audio y Video Bidireccional")
    
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.add(vbox)
    
        # Creamos los elementos de GStreamer
        self.pipeline = Gst.Pipeline()
        self.src = Gst.ElementFactory.make('filesrc', 'source')
        self.src.set_property('location', 'coco.mp4')
        self.vidconvert = Gst.ElementFactory.make('videoconvert', 'vidconvert')
        self.vidcaps = Gst.ElementFactory.make('capsfilter', 'vidcaps')
        self.vidcaps.set_property('caps', Gst.Caps.from_string('video/x-raw, width=640, height=480'))
        self.videosink = Gst.ElementFactory.make('autovideosink', 'videosink')
        self.sink = Gst.ElementFactory.make('alsasink', 'audiosink')
        self.mute = False

        # Agregamos los elementos al pipeline
        self.pipeline.add(self.src)
        self.pipeline.add(self.vidconvert)
        self.pipeline.add(self.vidcaps)
        self.pipeline.add(self.videosink)
        self.pipeline.add(self.sink)
    
        # Conectamos los elementos
        self.src.link(self.vidconvert)
        self.vidconvert.link(self.vidcaps)
        self.vidcaps.link(self.videosink)
        self.src.link(self.sink)
    
        # Creamos un botón para mutear el audio
        self.mute_button = Gtk.Button(label="Mute Audio")
        self.mute_button.connect("clicked", self.on_mute_button_clicked)
        vbox.pack_start(self.mute_button, True, True, 0)
    
        # Iniciamos el pipeline
        self.pipeline.set_state(Gst.State.PLAYING)

win = MyApp()
win.connect("destroy", Gtk.main_quit)
win.show_all()
Gtk.main()
