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

try:
    import scipy.signal
    import scipy.fftpack
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


class BaseSpectro(WidgetNode):
    """
    Common basis for spectrogram viewers. They all require a transform to frequency representation.
    In this architecture this transform is done by workers that could be in other node groups, thus 
    releasing the gil.
    """

    def __init__(self, **kargs):
        WidgetNode.__init__(self, **kargs)
        
        self.mainlayout = QtGui.QHBoxLayout()
        self.setLayout(self.mainlayout)
        
        self.graphiclayout = pg.GraphicsLayoutWidget()
        self.mainlayout.addWidget(self.graphiclayout)

    def show_params_controller(self):
        self.params_controller.show()

    def _configure(self, with_user_dialog=True, nodegroup_friends=None):
        self.with_user_dialog = with_user_dialog
        self.nodegroup_friends = nodegroup_friends
        self.local_workers = self.nodegroup_friends is None

    def _initialize(self, ):
        self.sample_rate = sr = self.input.params['sample_rate']
        d0, d1 = self.input.params['shape']
        if self.input.params['timeaxis']==0:
            self.nb_channel = d1
        else:
            self.nb_channel = d0
        
        # create proxy input to ensure sharedarray with time axis 1
        if self.input.params['transfermode'] == 'sharedarray' and self.input.params['timeaxis'] == 1:
            self.conv = None
        else:
            # if input is not transfermode creat a proxy
            if self.local_workers:
                self.conv = StreamConverter()
            else:
                ng = self.nodegroup_friends[-1]
                self.conv = ng.create_node('StreamConverter')
                self.conv.ng_proxy = ng
            self.conv.configure()

            # the inputstream is not needed except for parameters
            input_spec = dict(self.input.params)
            self.conv.input.connect(input_spec)
            
            if self.input.params['timeaxis']==0:
                new_shape = (d1, d0)
            else:
                new_shape = (d0, d1)
            self.conv.output.configure(protocol='tcp', interface='127.0.0.1', port='*', dtype='float32',
                   transfermode='sharedarray', streamtype='analogsignal', shape=new_shape, timeaxis=1, 
                   compression='', scale=None, offset=None, units='',
                   sharedarray_shape=(self.nb_channel, int(sr*self.max_xsize)), ring_buffer_method = 'double',
                   )
            self.conv.initialize()
            
        self.workers = []
        self.input_maps = []

        self.global_poller = ThreadPollInput(input_stream=self.input)
        self.global_timer = QtCore.QTimer(interval=500)
        self.global_timer.timeout.connect(self.compute_maps)
        
        if not self.local_workers:
            self.map_pollers = []
        
        for i in range(self.nb_channel):
            
            # create worker
            if self.local_workers:
                worker = self.worker_cls()
            else:
                ng = self.nodegroup_friends[i%max(len(self.nodegroup_friends)-1, 1)]
                worker = ng.create_node('TimeFreqWorker')
                worker.ng_proxy = ng
            worker.configure(max_xsize=self.max_xsize, channel=i, local=self.local_workers)
            worker.input.connect(self.conv.output)
            if self.local_workers:
                protocol = 'inproc'
            else:
                protocol = 'tcp'
            worker.output.configure(protocol=protocol, transfermode='plaindata')
            worker.initialize()
            self.workers.append(worker)
            
            # socket stream for maps from worker
            input_map = InputStream()
            stream_spec = dict(worker.output.params)
            input_map.connect(worker.output)
            self.input_maps.append(input_map)
            if self.local_workers:
                worker.wt_map_done.connect(self.on_new_map_local)
            else:
                poller = ThreadPollInput(input_stream=input_map)
                poller.new_data.connect(self.on_new_map_socket)
                poller.chan = i
                self.map_pollers.append(poller)
        
        # This is used to diffred heavy action whena changing params (setting plots, compute wavelet, ...)
        # this avoid overload on CPU if multiple changes occurs in a short time
        self.mutex_action = Mutex()
        self.actions = OrderedDict([(self.create_grid, False),
                                                    (self.initialize_freq_repr, False),
                                                    (self.initialize_plots, False),
                                                    ])
        self.timer_action = QtCore.QTimer(singleShot=True, interval=300)
        self.timer_action.timeout.connect(self.apply_actions)
        
        # Create parameters
        all = []
        for i in range(self.nb_channel):
            name = 'Signal{}'.format(i)
            all.append({'name': name, 'type': 'group', 'children': self._default_by_channel_params})
        self.by_channel_params = pg.parametertree.Parameter.create(name='AnalogSignals', type='group', children=all)
        self.params = pg.parametertree.Parameter.create(name='Global options',
                                                    type='group', children=self._default_params)
        self.all_params = pg.parametertree.Parameter.create(name='all param',
                                    type='group', children=[self.params,self.by_channel_params])
        self.params.param('xsize').setLimits([16./sr, self.max_xsize*.95]) 
        self.all_params.sigTreeStateChanged.connect(self.on_param_change)
        
        if self.with_user_dialog:
            self.params_controller = self.controller_cls(parent=self, viewer=self)
            self.params_controller.setWindowFlags(QtCore.Qt.Window)
        else:
            self.params_controller = None
        
        self.create_grid()
        self.initialize_freq_repr()
        self.initialize_plots()

    def _start(self):
        self.global_poller.start()
        self.global_timer.start()
        for worker in self.workers:
            worker.start()
        if not self.local_workers:
            for i in range(self.nb_channel):
                self.map_pollers[i].start()
        self.conv.start()
    
    def _stop(self):
        self.global_timer.stop()
        self.global_poller.stop()
        self.global_poller.wait()
        for worker in self.workers:
            worker.stop()
        if not self.local_workers:
            for i in range(self.nb_channel):
                self.map_pollers[i].stop()
                self.map_pollers[i].wait()
        self.conv.stop()
    
    def _close(self):
        if self.running():
            self.stop()
        if self.with_user_dialog:
            self.params_controller.close()
        for worker in self.workers:
            worker.close()
        self.conv.close()
        if not self.local_workers:
            # remove from NodeGroup
            self.conv.ng_proxy.delete_node(self.conv.name)
            for worker in self.workers:
                worker.ng_proxy.delete_node(worker.name)
    
    def create_grid(self):
        color = self.params['background_color']
        self.graphiclayout.clear()
        self.plots = [None] * self.nb_channel
        self.images = [None] * self.nb_channel
        r,c = 0,0
        nb_visible =sum(self.by_channel_params.children()[i]['visible'] for i in range(self.nb_channel)) 
        rowspan = self.params['nb_column']
        colspan = nb_visible//self.params['nb_column']
        self.graphiclayout.ci.currentRow = 0
        self.graphiclayout.ci.currentCol = 0        
        for i in range(self.nb_channel):
            if not self.by_channel_params.children()[i]['visible']: continue

            viewBox = MyViewBox()
            if self.with_user_dialog:
                viewBox.doubleclicked.connect(self.show_params_controller)
            viewBox.gain_zoom.connect(self.clim_zoom)
            viewBox.xsize_zoom.connect(self.xsize_zoom)
            
            plot = pg.PlotItem(viewBox=viewBox)
            plot.hideButtons()
            plot.showAxis('left', self.params['show_axis'])
            plot.showAxis('bottom', self.params['show_axis'])

            self.graphiclayout.ci.layout.addItem(plot, r, c)  # , rowspan, colspan)
            if r not in self.graphiclayout.ci.rows:
                self.graphiclayout.ci.rows[r] = {}
            self.graphiclayout.ci.rows[r][c] = plot
            self.graphiclayout.ci.items[plot] = [(r,c)]
            self.plots[i] = plot
            c+=1
            if c==self.params['nb_column']:
                c=0
                r+=1

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


class QSpectrogram(BaseSpectro):
    """
    Class for visualizing the frequency spectrogram with a Fourier transform.
    
    It proposes three views :  
    * Classical dynamic linear spectrogram (view_mode = 'line')
    * The same one but the previous lines are visible and transparent, smoothly vanishing (view_mode = 'blur')
    * Time-Frequency map (view_mode = 'time_freq')

    This class internally uses one worker per channel, which allows
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
    
    QSpectrogram can be configured on the fly by changing QSpectrogram.params and 
    QSpectrogram.by_channel_params. By default, double-clicking on the viewer 
    will open a GUI dialog for these parameters.
    
    
    Usage::
    
        viewer = QSpectrogram()
        viewer.configure(with_user_dialog=True, nodegroup_friends=None, view_mode='blur')
        viewer.input.connect(somedevice.output)
        viewer.initialize()
        viewer.show()
        viewer.start()
        
        viewer.params['nb_column'] = 4
        viewer.params['refresh_interval'] = 1000
    
    """
    pass

