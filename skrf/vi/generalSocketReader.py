#       generalSocketReader.py
#
#       Copyright 2011 alex arsenovic <arsenovic@virginia.edu>
#
#       This program is free software; you can redistribute it and/or modify
#       it under the terms of the GNU General Public License as published by
#       the Free Software Foundation; either version 2 of the License, or
#       (at your option) any later version.
#
#       This program is distributed in the hope that it will be useful,
#       but WITHOUT ANY WARRANTY; without even the implied warranty of
#       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#       GNU General Public License for more details.
#
#       You should have received a copy of the GNU General Public License
#       along with this program; if not, write to the Free Software
#       Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#       MA 02110-1301, USA.



import socket
import numpy as npy
from time import sleep
class GeneralSocketReader:
    '''
    A general class which wraps a socket and has a simple data query
    function, implemented by the property data_point.

    this was made as a way to interface a piece of hardware which did
    not support     gpib.  is useful for general interfacing  of
    non-standard hardware or software.

    example usage:
            gsr = generalSocketRead()
            gsr.connect('127.0.0.1',1111)
            gsr.data_point  # implicityly calls send() then receive()
    '''
    def __init__(self, sock=None, sample_rate=2.5, avg_len=1,\
            query_string = '1', msg_len =1e3):
        '''
        takes:
                sock: socket type (defaults to None and generates a new socket)
                query_string: string sent during send() command
                msg_len: length of recv buffer used in receive() command

        '''
        if sock is None:
            self.sock = socket.socket(
                    socket.AF_INET, socket.SOCK_STREAM)
        else:
            self.sock = sock
        self.query_string = query_string
        self.msg_len = int(msg_len)
        self.sample_rate = sample_rate*1.0
        self.avg_len = avg_len


    def connect(self, host, port):
        self.sock.connect((host, port))

    def close(self):
        self.sock.close()

    def send(self, data):
        self.sock.send(data)

    def receive(self):
        data = self.sock.recv(self.msg_len)
        return data

    @property
    def data(self):
        '''tmp = []
        for n in range(self.avg_len):
                sleep(1./self.sample_rate)
                self.send(self.query_string)
                tmp.append(float(self.receive()))
        return npy.mean(tmp)
        '''
        self.send(self.query_string)
        return float(self.receive())
