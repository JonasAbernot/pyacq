# -*- coding: utf-8 -*-
"""
Module for recording streams to disk.

"""

import numpy as np
import zmq
import threading
import io
import msgpack
import os
import json
import time


from ..version import version as pyacq_version

class RawDataRecording:
    """
    Raw data recording one file per stream support only:
        * AnalogSignalSharedMemStream and DigitalSignalSharedMemStream
    
    
    """
    def __init__(self, streams, dirname, dtype_sig = np.float32,
                                    annotations = { },
                                    
                                    ):
        self.streams = streams
        self.dirname = dirname
        
        self.dtype_sig = dtype_sig
        self.annotations = annotations
        
        
    def start(self):

        self.running = True
        
        # header in json
        info = { }
        info['pyacq_version'] = pyacq_version
        info['streams'] = [ ]
        for stream in self.streams:
            infostream = { }
            infostream.update(stream._params)
            
            infostream['stream_type'] = type(stream).__name__
            
            if infostream['stream_type'] == 'AnalogSignalSharedMemStream':
                infostream['dtype'] = str(np.dtype(self.dtype_sig).name)
            
            for e in ['shared_array', ]:
                if e in infostream:
                    infostream.pop(e)
            info['streams'].append(infostream)
        info['annotations'] = self.annotations
        
        info_file = io.open(os.path.join(self.dirname, 'info.json'), mode = 'w', encoding = 'utf8')
        info_file.write(json.dumps(info, sort_keys=True, indent=4, separators=(',', ': '), ensure_ascii = False, encoding =  'utf8'))
        info_file.close()
        
        # one thread per stream
        self.threads = [ ]
        self.files = [ ]
        context = zmq.Context()
        for stream in self.streams:
            socket = context.socket(zmq.SUB)
            socket.setsockopt(zmq.SUBSCRIBE,'')
            socket.connect("tcp://localhost:{}".format(stream['port']))
            
            f = io.open(os.path.join(self.dirname, stream.name+'.raw'), mode = 'wb')
            self.files.append(f)
            
            callback = 'rec_loop_'+type(stream).__name__
            if hasattr(self, callback):
                func = getattr(self, callback)
                thread = threading.Thread(target = func, args = (socket,stream, f))
                self.threads.append(thread)
                thread.start()
        
    
    def rec_loop_AnalogSignalSharedMemStream(self, socket, stream, file):
        last_pos = None
        np_array = stream['shared_array'].to_numpy_array()
        half_size = np_array.shape[1]/2
        while self.running:
            message = socket.recv()
            pos = msgpack.loads(message)
            if last_pos is None:
                last_pos = pos
                continue
            new = (pos-last_pos)
            if new>half_size:
                print 'ERROR MISSEDE packet'
                # FIXME : what to do
                new = half_size
            head = pos%half_size+half_size
            tail = head - new
            file.write(np_array[:, tail:head].transpose().astype(self.dtype_sig).tostring())
            last_pos = pos

    def rec_loop_DigitalSignalSharedMemStream(self, socket, stream, file):
        last_pos = None
        np_array = stream['shared_array'].to_numpy_array()
        half_size = np_array.shape[1]/2
        while self.running:
            message = socket.recv()
            pos = msgpack.loads(message)
            if last_pos is None:
                last_pos = pos
                continue
            new = (pos-last_pos)
            if new>half_size:
                print 'ERROR MISSEDE packet'
                # FIXME : what to do
                new = half_size
            head = pos%half_size+half_size
            tail = head - new
            file.write(np_array[:, tail:head].transpose().tostring())
            last_pos = pos

    def rec_loop_AsynchronusEventStream(self, socket, stream, file):
        while self.running:
            if socket.poll(timeout = 100):
                
                message = socket.recv()
                t = time.time()
                print 'ici', t, message
                file.write(buffer(np.float64(t)))
                file.write(message)
            

        
    


    def stop(self):
        self.running = False
        for thread in self.threads:
            thread.join()
        for f in self.files:
            f.close()
        #~ print 'rec stopped'
        


