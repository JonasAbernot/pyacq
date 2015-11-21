from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
from pyqtgraph.util.mutex import Mutex
import vispy.color

import sys
import numpy as np
import weakref
import time
from collections import OrderedDict

from ..core import (WidgetNode, Node, register_node_type, InputStream, OutputStream,
        ThreadPollInput, StreamConverter, StreamSplitter)

from .qoscilloscope import MyViewBox
from .qspectrogram import BaseSpectro

try:
    import scipy.signal
    import scipy.fftpack
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


default_params = [
        {'name': 'xsize', 'type': 'float', 'value': 10., 'step': 0.1, 'limits': (.1, 60)},
        {'name': 'nb_column', 'type': 'int', 'value': 1},
        {'name': 'background_color', 'type': 'color', 'value': 'k'},
        #~ {'name': 'colormap', 'type': 'list', 'value': 'hot', 'values' : ['hot', 'coolwarm', 'ice', 'grays', ] },
        {'name': 'colormap', 'type': 'list', 'value': 'hot', 'values': list(vispy.color.get_colormaps().keys())},
        {'name': 'refresh_interval', 'type': 'int', 'value': 500, 'limits':[5, 1000]},
        {'name': 'mode', 'type': 'list', 'value': 'scroll', 'values': ['scan', 'scroll']},
        {'name': 'show_axis', 'type': 'bool', 'value': False},
        # ~ {'name': 'display_labels', 'type': 'bool', 'value': False }, #TODO when title
        {'name': 'timefreq', 'type': 'group', 'children': [
                        {'name': 'f_start', 'type': 'float', 'value': 3., 'step': 1.},
                        {'name': 'f_stop', 'type': 'float', 'value': 90., 'step': 1.},
                        {'name': 'deltafreq', 'type': 'float', 'value': 3., 'step': 1., 'limits': [0.1, 1.e6]},
                        {'name': 'f0', 'type': 'float', 'value': 2.5, 'step': 0.1},
                        {'name': 'normalisation', 'type': 'float', 'value': 0., 'step': 0.1},]}
    ]

default_by_channel_params = [ 
                {'name': 'visible', 'type': 'bool', 'value': True},
                {'name': 'clim', 'type': 'float', 'value': 1.},
            ]


class QTimeFreq(BaseSpectro):
    """
    Class for visualizing the frequency spectrogram with a Morlet continuous
    wavelet transform.
    
    This allows better visualization than the standard FFT spectrogram because
    it provides better temporal resolution for high-frequency signals without
    sacrificing frequency resolution for low-frequency signals.
    See https://en.wikipedia.org/wiki/Morlet_wavelet
    
    This class internally uses one TimeFreqWorker per channel, which allows
    multiple signals to be transformed in parallel.
        
    The node operates in one of 2 modes:
    
    * Each TimeFreqWorker lives in the same QApplication as the QTimeFreq node
      (nodegroup_friends=None).
    * Each TimeFreqWorker is spawned in another NodeGroup to distribute the
      load (nodegroup_friends=[some_list_of_nodegroup]).
    
    This viewer needs manual tuning for performance: small refresh_interval, 
    high number of freqs, hight f_stop, and high xsize can all lead to heavy
    CPU load.
    
    This node requires its input stream to use:
    
    * ``transfermode==sharedarray``
    * ``timeaxis==1``
    
    If the input stream does not meet these requirements, then a StreamConverter
    will be created to proxy the input.
    
    QTimeFreq can be configured on the fly by changing QTimeFreq.params and 
    QTimeFreq.by_channel_params. By default, double-clicking on the viewer 
    will open a GUI dialog for these parameters.
    
    
    Usage::
    
        viewer = QTimeFreq()
        viewer.configure(with_user_dialog=True, nodegroup_friends=None)
        viewer.input.connect(somedevice.output)
        viewer.initialize()
        viewer.show()
        viewer.start()
        
        viewer.params['nb_column'] = 4
        viewer.params['refresh_interval'] = 1000
    
    """
    
    _input_specs = {'signal': dict(streamtype='signals', shape=(-1), )}
    
    _default_params = default_params
    _default_by_channel_params = default_by_channel_params

    def __init__(self, **kargs):
        BaseSpectro.__init__(self, **kargs)
        self.worker_cls = TimeFreqWorker
        self.controller_cls = TimeFreqController

    def _configure(self,max_xsize=60., **kargs):
        BaseSpectro._configure(self,**kargs)
        self.max_xsize = max_xsize
    
    def initialize_freq_repr(self):
        tfr_params = self.params.param('timefreq')
        
        # we take sample_rate = f_stop*4 or (original sample_rate)
        if tfr_params['f_stop']*4 < self.sample_rate:
            sub_sample_rate = tfr_params['f_stop']*4
        else:
            sub_sample_rate = self.sample_rate
        
        # this try to find the best size to get a timefreq of 2**N by changing
        # the sub_sample_rate and the sig_chunk_size
        self.wanted_size = self.params['xsize']
        self.len_wavelet = l = int(2**np.ceil(np.log(self.wanted_size*sub_sample_rate)/np.log(2)))
        self.sig_chunk_size = self.wanted_size*self.sample_rate
        self.downsample_factor = int(np.ceil(self.sig_chunk_size/l))
        self.sig_chunk_size = self.downsample_factor*l
        self.sub_sample_rate = self.sample_rate/self.downsample_factor
        self.plot_length = int(self.wanted_size*sub_sample_rate)
        
        self.wavelet_fourrier = generate_wavelet_fourier(self.len_wavelet, tfr_params['f_start'], tfr_params['f_stop'],
                            tfr_params['deltafreq'], self.sub_sample_rate, tfr_params['f0'], tfr_params['normalisation'])
        
        if self.downsample_factor>1:
            self.filter_b = scipy.signal.firwin(9, 1. / self.downsample_factor, window='hamming')
            self.filter_a = np.array([1.])
        else:
            self.filter_b = None
            self.filter_a = None
        
        for worker in self.workers:
            worker.on_fly_change_wavelet(wavelet_fourrier=self.wavelet_fourrier, downsample_factor=self.downsample_factor,
                        sig_chunk_size=self.sig_chunk_size, plot_length=self.plot_length, filter_a=self.filter_a, filter_b=self.filter_b)
        
        for input_map in self.input_maps:
            input_map.params['shape'] = (self.plot_length, self.wavelet_fourrier.shape[1])
            input_map.params['sample_rate'] = sub_sample_rate
    
    def initialize_plots(self):
        N = 512
        cmap = vispy.color.get_colormap(self.params['colormap'])
        self.lut = (255*cmap.map(np.arange(N)[:,None]/float(N))).astype('uint8')
        
        tfr_params = self.params.param('timefreq')
        for i in range(self.nb_channel):
            if self.by_channel_params.children()[i]['visible']:
                for item in self.plots[i].items:
                    # remove old images
                    self.plots[i].removeItem(item)
                
                clim = self.by_channel_params.children()[i]['clim']
                f_start, f_stop = tfr_params['f_start'], tfr_params['f_stop']
                
                image = pg.ImageItem()
                image.setImage(np.zeros((self.plot_length,self.wavelet_fourrier.shape[1])), lut=self.lut, levels=[0,clim])
                self.plots[i].addItem(image)
                image.setRect(QtCore.QRectF(-self.wanted_size, f_start,self.wanted_size, f_stop-f_start))
                self.plots[i].setXRange(-self.wanted_size, 0.)
                self.plots[i].setYRange(f_start, f_stop)
                self.images[i] =image
    
    def on_param_change(self, params, changes):
        for param, change, data in changes:
            if change != 'value': continue
            # immediate action
            if param.name()=='background_color':
                color = data
                for graphicsview in self.graphicsviews:
                    if graphicsview is not None:
                        graphicsview.setBackground(color)
            if param.name()=='refresh_interval':
                self.global_timer.setInterval(data)
            if param.name()=='clim':
                i = self.by_channel_params.children().index(param.parent())
                clim = param.value()
                if self.images[i] is not None:
                    self.images[i].setImage(self.images[i].image, lut=self.lut, levels=[0,clim])
            if param.name()=='show_axis':
                for plot in self.plots:
                    if plot is not None:
                        plot.showAxis('left', data)
                        plot.showAxis('bottom', data)                        
            
            # difered action delayed with timer
            with self.mutex_action:
                if param.name()=='xsize':
                    self.actions[self.initialize_freq_repr] = True
                    self.actions[self.initialize_plots] = True
                if param.name()=='colormap':
                    self.actions[self.initialize_plots] = True
                if param.name()=='nb_column':
                    self.actions[self.create_grid] = True
                    self.actions[self.initialize_plots] = True
                if param.name() in ('f_start', 'f_stop', 'deltafreq', 'f0', 'normalisation'):
                    self.actions[self.initialize_freq_repr] = True
                    self.actions[self.initialize_plots] = True
                if param.name()=='visible':
                    self.actions[self.create_grid] = True
                    self.actions[self.initialize_plots] = True
        
        with self.mutex_action:
            if not self.timer_action.isActive() and any(self.actions.values()):
                self.timer_action.start()
    
    def apply_actions(self):
        with self.mutex_action:
            if self.running():
                self.global_timer.stop()
            for action, do_it in self.actions.items():
                if do_it:
                    action()
            for action in self.actions:
                self.actions[action] = False
            if self.running():
                self.global_timer.start()
    
    def compute_maps(self):
        head = int(self.global_poller.pos())
        for i in range(self.nb_channel):
            if self.by_channel_params.children()[i]['visible']:
                if self.local_workers:
                    self.workers[i].compute_one_map(head)
                else:
                    self.workers[i].compute_one_map(head, _sync=False)
    
    def on_new_map_local(self, chan):
        head, wt_map = self.input_maps[chan].recv()
        self.update_image(chan, head, wt_map)
    
    def on_new_map_socket(self, head, wt_map):
        chan = self.sender().chan
        self.update_image(chan, head, wt_map)

    def update_image(self, chan, head, wt_map):
        if self.images[chan] is None: return
        if self.params['mode']=='scroll':
            self.images[chan].updateImage(wt_map)
        elif self.params['mode'] =='scan':
            ind = (head//self.downsample_factor)%self.plot_length+1
            wt_map = np.concatenate([wt_map[-ind:, :], wt_map[:-ind, :]], axis=0)
            self.images[chan].updateImage(wt_map)
    
    def clim_zoom(self, factor):
        for i, p in enumerate(self.by_channel_params.children()):
            p.param('clim').setValue(p.param('clim').value()*factor)
        
    def xsize_zoom(self, xmove):
        factor = xmove/100.
        newsize = self.params['xsize']*(factor+1.)
        limits = self.params.param('xsize').opts['limits']
        if newsize>limits[0] and newsize<limits[1]:
            self.params['xsize'] = newsize    
    
    def auto_clim(self, identic=True):
        if identic:
            all = []
            for i, p in enumerate(self.by_channel_params.children()):
                if p.param('visible').value():
                    all.append(np.max(self.images[i].image))
            clim = np.max(all)*1.1
            for i, p in enumerate(self.by_channel_params.children()):
                if p.param('visible').value():
                    p.param('clim').setValue(float(clim))
        else:
            for i, p in enumerate(self.by_channel_params.children()):
                if p.param('visible').value():
                    clim = np.max(self.images[i].image)*1.1
                    p.param('clim').setValue(float(clim))

register_node_type(QTimeFreq)


def generate_wavelet_fourier(len_wavelet, f_start, f_stop, deltafreq, sample_rate, f0, normalisation):
    """
    Compute the wavelet coefficients at all scales and compute its Fourier transform.
    
    Parameters
    ----------
    len_wavelet : int
        length in samples of the wavelet window
    f_start: float
        First frequency in Hz
    f_stop: float
        Last frequency in Hz
    deltafreq : float
        Frequency interval in Hz
    sample_rate : float
        Sample rate in Hz
    f0 : float
    normalisation : float
    
    Returns:
    -------
    
    wf : array
        Fourier transform of the wavelet coefficients (after weighting).
        Axis 0 is time; axis 1 is frequency.
    """
    # compute final map scales
    scales = f0/np.arange(f_start,f_stop,deltafreq)*sample_rate
    
    # compute wavelet coeffs at all scales
    xi=np.arange(-len_wavelet/2.,len_wavelet/2.)
    xsd = xi[:,np.newaxis] / scales
    wavelet_coefs=np.exp(complex(1j)*2.*np.pi*f0*xsd)*np.exp(-np.power(xsd,2)/2.)

    weighting_function = lambda x: x**(-(1.0+normalisation))
    wavelet_coefs = wavelet_coefs*weighting_function(scales[np.newaxis,:])

    # Transform the wavelet into the Fourier domain
    wf=scipy.fftpack.fft(wavelet_coefs,axis=0)
    wf=wf.conj()
    
    return wf



class ComputeThread(QtCore.QThread):
    """
    Worker thread used internally by TimeFreqWorker.
    """
    def __init__(self, in_stream, out_stream, channel, local, parent=None):
        QtCore.QThread.__init__(self, parent)
        self.in_stream = weakref.ref(in_stream)
        self.out_stream = weakref.ref(out_stream)
        self.channel = channel
        self.local = local
        
        self.worker_params = None

    def run(self):
        if self.worker_params is None: 
            return
        head = self.head
        
        downsample_factor = self.worker_params['downsample_factor']
        sig_chunk_size = self.worker_params['sig_chunk_size']
        filter_a = self.worker_params['filter_a']
        filter_b = self.worker_params['filter_b']
        wavelet_fourrier = self.worker_params['wavelet_fourrier']
        plot_length = self.worker_params['plot_length']
        
        #~ t1 = time.time()
        
        if downsample_factor>1:
            head = head - head%downsample_factor
        full_arr = self.in_stream().get_array_slice(head, sig_chunk_size)[self.channel, :]
        
        #~ t2 = time.time()
        
        if downsample_factor>1:
            small_arr = scipy.signal.filtfilt(filter_b, filter_a, full_arr)
            small_arr =small_arr[::downsample_factor].copy()  # to ensure continuity
        else:
            small_arr = full_arr
        
        small_arr_f=scipy.fftpack.fft(small_arr)
        if small_arr_f.shape[0] != wavelet_fourrier.shape[0]: return
        wt_tmp=scipy.fftpack.ifft(small_arr_f[:,np.newaxis]*wavelet_fourrier,axis=0)
        wt = scipy.fftpack.fftshift(wt_tmp,axes=[0])
        wt = np.abs(wt).astype('float32')
        wt = wt[-plot_length:]
        
        #~ self.last_wt_map = wt
        self.out_stream().send(head, wt)
        #~ t3 = time.time()
        
        # print('compute', self.channel,  t2-t1, t3-t2, t3-t1, QtCore.QThread.currentThreadId())



class TimeFreqWorker(Node, QtCore.QObject):
    """
    TimeFreqWorker is a Node that computes the frequency spectrogram with a 
    Morlet continuous wavelet transform.
    
    This allows better visualization than the standard FFT spectrogram because
    it provides better temporal resolution for high-frequency signals without
    sacrificing frequency resolution for low-frequency signals.
    See https://en.wikipedia.org/wiki/Morlet_wavelet

    The computation is quite heavy: Each signal chunk is first downsampled 
    (with a filtfilt first), then convolved (using FFT method) with one wavelet
    per frequency to be analyzed.

    For visualization of this analysis, use QTimeFreq.
    """
    _input_specs = {'signal': dict(streamtype='signals', transfermode='sharedarray', timeaxis=1, ring_buffer_method='double')}
    _output_specs = {'timefreq': dict(streamtype='image', dtype='float32')}
    
    wt_map_done = QtCore.pyqtSignal(int)
    def __init__(self, **kargs):
        parent = kargs.pop('parent', None)
        QtCore.QObject.__init__(self, parent)
        Node.__init__(self, **kargs)
        assert HAVE_SCIPY, "TimeFreqWorker node depends on the `scipy` package, but it could not be imported."
    
    def _configure(self, max_xsize=60., channel=None, local=True):
        self.max_xsize = max_xsize
        self.channel = channel
        self.local = local
    
    def after_input_connect(self, inputname):
        assert len(self.input.params['shape']) == 2, 'Wrong shape: TimeFreqWorker'
        assert self.input.params['timeaxis'] == 1, 'Wrong timeaxis: TimeFreqWorker'
        assert self.input.params['transfermode'] == 'sharedarray', 'Wrong shape: sharedarray'
        
        
    def _initialize(self):
        self.sample_rate = sr = self.input.params['sample_rate']
        self.thread = ComputeThread(self.input, self.output, self.channel, self.local)
        self.thread.finished.connect(self.on_thread_done)

    def _start(self):
        pass
    
    def _stop(self):
        if self.thread.isRunning():
            self.thread.wait()
    
    def _close(self):
        pass
    
    #~ def on_fly_change_wavelet(self, wavelet_fourrier=None, downsample_factor=None, sig_chunk_size = None,
            #~ plot_length=None, filter_a=None, filter_b=None):
    def on_fly_change_wavelet(self, **worker_params):
        p = worker_params
        
        if not self.local:
            # with our RPC ndarray came from np.frombuffer
            # but scipy.signal.filtflt need b writtable so:
            p['filter_b'] = p['filter_b'].copy()
        
        p['out_shape'] = (p['plot_length'], p['wavelet_fourrier'].shape[1])
        self.output.params['shape'] = p['out_shape']
        self.output.params['sample_rate'] = self.sample_rate/p['downsample_factor']
        
        self.worker_params = worker_params
    
    def on_thread_done(self):
        self.thread.wait()
        self.wt_map_done.emit(self.channel)
        self.thread.workers_params = None
    
    def compute_one_map(self, head):
        assert self.running(), 'TimeFreqWorker is not running'
        if self.thread.isRunning():
            return
        if self.closed():
            return
        self.thread.worker_params = self.worker_params
        self.thread.head = head
        self.thread.start()

register_node_type(TimeFreqWorker)




class TimeFreqController(QtGui.QWidget):
    """
    GUI controller for QTimeFreq.
    """
    def __init__(self, parent=None, viewer=None):
        QtGui.QWidget.__init__(self, parent)
        
        self._viewer = weakref.ref(viewer)
        
        # layout
        self.mainlayout = QtGui.QVBoxLayout()
        self.setLayout(self.mainlayout)
        t = 'Options for {}'.format(self.viewer.name)
        self.setWindowTitle(t)
        self.mainlayout.addWidget(QtGui.QLabel('<b>'+t+'<\b>'))
        
        h = QtGui.QHBoxLayout()
        self.mainlayout.addLayout(h)

        self.tree_params = pg.parametertree.ParameterTree()
        self.tree_params.setParameters(self.viewer.params, showTop=True)
        self.tree_params.header().hide()
        h.addWidget(self.tree_params)

        self.tree_by_channel_params = pg.parametertree.ParameterTree()
        self.tree_by_channel_params.header().hide()
        h.addWidget(self.tree_by_channel_params)
        self.tree_by_channel_params.setParameters(self.viewer.by_channel_params, showTop=True)

        v = QtGui.QVBoxLayout()
        h.addLayout(v)
        
        if self.viewer.nb_channel>1:
            v.addWidget(QtGui.QLabel('<b>Select channel...</b>'))
            names = [p.name() for p in self.viewer.by_channel_params]
            self.qlist = QtGui.QListWidget()
            v.addWidget(self.qlist, 2)
            self.qlist.addItems(names)
            self.qlist.setSelectionMode(QtGui.QAbstractItemView.ExtendedSelection)
            
            for i in range(len(names)):
                self.qlist.item(i).setSelected(True)            
            v.addWidget(QtGui.QLabel('<b>and apply...<\b>'))
        
        but = QtGui.QPushButton('set visble')
        v.addWidget(but)
        but.clicked.connect(self.on_set_visible)
        
        but = QtGui.QPushButton('Automatic clim (same for all)')
        but.clicked.connect(lambda: self.auto_clim(identic=True))
        v.addWidget(but)

        but = QtGui.QPushButton('Automatic clim (independant)')
        but.clicked.connect(lambda: self.auto_clim(identic=False))
        v.addWidget(but)
        
        v.addWidget(QtGui.QLabel(self.tr('<b>Clim change (mouse wheel on graph):</b>'),self))
        h = QtGui.QHBoxLayout()
        v.addLayout(h)
        for label, factor in [('--', 1./10.), ('-', 1./1.3), ('+', 1.3), ('++', 10.),]:
            but = QtGui.QPushButton(label)
            but.factor = factor
            but.clicked.connect(self.clim_zoom)
            h.addWidget(but)
    
    @property
    def viewer(self):
        return self._viewer()

    @property
    def selected(self):
        selected = np.ones(self.viewer.nb_channel, dtype=bool)
        if self.viewer.nb_channel>1:
            selected[:] = False
            selected[[ind.row() for ind in self.qlist.selectedIndexes()]] = True
        return selected
    
    def on_set_visible(self):
        # apply
        visibles = self.selected
        for i,param in enumerate(self.viewer.by_channel_params.children()):
            param['visible'] = visibles[i]

    def auto_clim(self, identic=True):
        self.viewer.auto_clim(identic=identic)

    def clim_zoom(self):
        factor = self.sender().factor
        for i, p in enumerate(self.viewer.by_channel_params.children()):
            p.param('clim').setValue(p.param('clim').value()*factor)

