import ctypes
import numpy
from warnings import warn


#### Constants

from daqmxconstants import *

DAQmx_Val_Cfg_Default = -1

TASK_TYPE_INPUT  = 1
TASK_TYPE_OUTPUT = 1

terminal_config_names = {'rse': DAQmx_Val_RSE,
                         'nrse': DAQmx_Val_NRSE,
                         'diff': DAQmx_Val_Diff,
                         'pseudo-diff': DAQmx_Val_PseudoDiff}

sample_mode_names = {'continuous': DAQmx_Val_ContSamps,
                     'finite': DAQmx_Val_FiniteSamps,
                     'hardware': DAQmx_Val_HWTimedSinglePoint}


#### Types

nidaq      = ctypes.windll.nicaiu
int32      = ctypes.c_long
uint32     = ctypes.c_ulong
uint64     = ctypes.c_ulonglong
float64    = ctypes.c_double
bool32     = ctypes.c_ulong
TaskHandle = uint32 # On 32-bit machines


#### Error handling

def get_error_string(err):
    '''Retrieve the string associated with the error.'''
    if err == 0:
        return ''
    else:
        buf_size = 100
        buf = ctypes.create_string_buffer('\000' * buf_size)
        nidaq.DAQmxGetErrorString(err, ctypes.byref(buf), buf_size)
        return buf.value

def get_extended_error_info():
    '''Retrieve the string associated with the error.'''
    buf_size = 1024
    buf = ctypes.create_string_buffer('\000' * buf_size)
    nidaq.DAQmxGetExtendedErrorInfo(ctypes.byref(buf), buf_size)
    return buf.value

class DAQmxError(Exception):
    '''Error specific to the DAQmx functions.'''
    def __init__(self, err, desc=None):
      Exception.__init__(self)
      if type(err) == int:
        self.err  = get_error_string(err)
        self.desc = get_extended_error_info()
      else:
        self.err  = err
        self.desc = desc

    def __str__(self):
      if self.desc:
        return '%s: %s' % (self.err, self.desc)
      else:
        return self.err

class DAQmxWarning(UserWarning):
    '''Warning specific to the DAQmx functions.'''
    def __init__(self, err):
      UserWarning.__init__(self)
      if type(err) == int:
        self.err = get_error_string(err)
      else:
        self.err = err

    def __str__(self):
      return self.err

def CHK(err):
    '''a simple error checking routine'''
    if err < 0:
        raise DAQmxError(err)
    elif err > 0:
        warn(err, DAQmxWarning)
    else:
        pass


#### Utilities

def get_string_value(object, nidaqfunc, buffer_size=1024):
    buffer = ctypes.create_string_buffer(buffer_size)
    CHK(nidaqfunc(object, buffer, buffer_size))
    return buffer.value

def get_list_value(object, nidaqfunc):
    return map(str.strip, get_string_value(object,nidaqfunc).split(','))

def get_array_value(object, nidaqfunc, size=100, dtype='float64'):
    buffer = create_contiguous_buffer(size, dtype)
    CHK(nidaqfunc(object, buffer.ctypes.data, size))
    return buffer

def get_value(object, nidaqfunc, type):
    try:
        if type == list:
            return get_list_value(object, nidaqfunc)
        elif type == numpy.ndarray:
            return get_array_value(object, nidaqfunc)
        elif type == str:
            return get_string_value(object, nidaqfunc)
        else:
            data = type(0)
            CHK(nidaqfunc(object, ctypes.byref(data)))
            return data.value
    except:
        None

def array_value_as_ranges(buffer):
    buffer = buffer[numpy.where(buffer != 0)]
    return [(buffer[n*2], buffer[n*2+1]) for n in range(len(buffer)//2)]

def set_list_value(object, nidaqfunc, value):
    set_value(object, nidaqfunc, ','.join(value))

def set_value(object, nidaqfunc, value):
    if type == list:
        set_list_value(object, nidaqfunc, value)
    else:
        CHK(nidaqfunc(object, value))

def create_contiguous_buffer(size, dtype='float64', attempts=10):
    '''Create a contiguous buffer of given size and datatype
    ['float64'].  If creation fails, it will try again up to specified
    number of attempts [10].'''
    
    for count in range(attempts):
        buffer = numpy.zeros(size, dtype)
        if buffer.flags['C_CONTIGUOUS']:
            break
        else:       
            raise RunTimeError('Could not allocate contiguous buffer of size %d in %d attempts.' % (size, attempts))

    return buffer


def connect_terminals(source, destination, inverted=False):
    '''Connect the specified terminals.'''
    CHK(nidaq.DAQmxConnectTerms(source,
                                destination,
                                DAQmx_Val_InvertPolarity if inverted else DAQmx_Val_DoNotInvertPolarity))

#### Classes

def norm_channel(channel, device = None, channel_type = None, direction = None):
    if type(channel) == str:
        if device is None:
            return channel
        else:
            return '%s/%s' % (device, channel.split('/')[-1])
    elif type(channel) == int and device is not None and channel_type is not None and direction is not None:
        channel_types = {'analog': 'a',
                         'a' : 'a',
                         'counter': 'ctr',
                         'ctr': 'ctr',
                         'digital': 'd',
                         'd': 'd'}
        channel_directions = {'input': 'i',
                              'i': 'i',
                              'output': 'o',
                              'o': 'o'}
        if channel_types[channel_type] == 'ctr':
            return '%s/%s%d' % (device, channel_types[channel_type], channel)
        else:
            return '%s/%s%s%d' % (device, channel_types[channel_type],
                                  channel_directions[direction],
                                  channel)
    else:
        raise ValueError('Insufficient information to normalize channel name.')

### Classes

class Device(object):
  def __init__(self, name='Dev1'):
    self.name = name.capitalize()

  ## Some properties

  product_type = property(fget =
      lambda(x): get_value(x.name,nidaq.DAQmxGetDevProductType, str),
      doc = 'Product type of this device')
  serial_number = property(fget =
      lambda(x): get_value(x.name,nidaq.DAQmxGetDevSerialNum, uint32),
      doc = 'Serial number of this device')

  analog_input_channels = property(fget =
      lambda(x): get_value(x.name,nidaq.DAQmxGetDevAIPhysicalChans, list),
      doc = 'List of physical analog input channels')
  analog_input_maximum_single_channel_rate = property(fget =
      lambda(x): get_value(x.name,nidaq.DAQmxGetDevAIMaxSingleChanRate, float64),
      doc = 'Maximum sample rate for single analog input channels')
  analog_input_maximum_multi_channel_rate = property(fget =
      lambda(x): get_value(x.name,nidaq.DAQmxGetDevAIMaxMultiChanRate, float64),
      doc = 'Maximum sample rate for multiple analog input channels')
  analog_input_voltage_ranges = property(fget =
      lambda(x): array_value_as_ranges(get_value(x.name,
        nidaq.DAQmxGetDevAIVoltageRngs, numpy.ndarray)),
      doc = 'Voltage ranges for analog input channels')

  analog_output_channels = property(fget =
      lambda(x): get_value(x.name,nidaq.DAQmxGetDevAOPhysicalChans, list),
      doc = 'List of physical analog output channels')
  analog_output_minimum_rate = property(fget =
      lambda(x): get_value(x.name,nidaq.DAQmxGetDevAOMinRate, float64),
      doc = 'Minimum sample rate for analog output channels')
  analog_output_maximum_rate = property(fget =
      lambda(x): get_value(x.name,nidaq.DAQmxGetDevAOMaxRate, float64),
      doc = 'Maximum sample rate for analog output channels')
  analog_output_voltage_ranges = property(fget =
      lambda(x): array_value_as_ranges(get_value(x.name,
        nidaq.DAQmxGetDevAOVoltageRngs, numpy.ndarray)),
      doc = 'Voltage ranges for analog output channels')

  digital_input_lines = property(fget =
      lambda(x): get_value(x.name, nidaq.DAQmxGetDevDILines, list),
      doc = 'List of digital input lines')
  digital_input_maximum_rate = property(fget =
      lambda(x): get_value(x.name, nidaq.DAQmxGetDevDIMaxRate, float64),
      doc = 'Maximum sample rate for digital input channels')
  digital_input_ports = property(fget =
      lambda(x): get_value(x.name, nidaq.DAQmxGetDevDIPorts, list),
      doc = 'List of digital input ports')

  digital_output_lines = property(fget =
      lambda(x): get_value(x.name, nidaq.DAQmxGetDevDOLines, list),
      doc = 'List of digital output lines')
  digital_output_maximum_rate = property(fget =
      lambda(x): get_value(x.name, nidaq.DAQmxGetDevDOMaxRate, float64),
      doc = 'Maximum sample rate for digital output channels')
  digital_output_ports = property(fget =
      lambda(x): get_value(x.name, nidaq.DAQmxGetDevDOPorts, list),
      doc = 'List of digital output ports')

  counter_input_channels = property(fget =
      lambda(x): get_value(x.name, nidaq.DAQmxGetDevCIPhysicalChans, list),
      doc = 'List of counter input channels')
  counter_input_maximum_size = property(fget =
      lambda(x): get_value(x.name, nidaq.DAQmxGetDevCIMaxSize, uint32),
      doc = 'Maximum size in bits of for the counter input channels')

  counter_output_channels = property(fget =
      lambda(x): get_value(x.name, nidaq.DAQmxGetDevCOPhysicalChans, list),
      doc = 'List of counter output channels')
  counter_output_maximum_size = property(fget =
      lambda(x): get_value(x.name, nidaq.DAQmxGetDevCOMaxSize, uint32),
      doc = 'Maximum size in bits of for the counter output channels')

  def reset(self):
    '''Reset the device.'''
    CHK(nidaq.DAQmxResetDevice(self.name))

  def has_channel(self, channel, channel_type=None, channel_direction=None):
    '''Check whether the given channel exists on this device'''
    result = False
    if type(channel) == int:
      channel = '%s%s%d' % (channel_types[channel_type],
          channel_directions[channel_direction])
    x = channel.split('/')
    channel = x[-1]
    if len(x) > 1 and x[0].capitalize() != self.name:
      result = False
    else:
      chans = {'ai': self.analog_input_channels,
          'ao': self.analog_output_channels}
      if int(channel[2:]) < len(chans[channel[0:2]]):
        result = True
      else:
        result = False
    return result

  def has_analog_input_channel(self, channel):
    '''Check whether the given analog input channel (may be an integer) exists on this device'''
    if type(channel) == int:
      return self.has_channel('ai%d' % channel)
    else:
      return self.has_channel(channel)

  def has_analog_output_channel(self, channel):
    '''Check whether the given analog output channel (may be an integer) exists on this device'''
    if type(channel) == int:
      return self.has_channel('ao%d' % channel)
    else:
      return self.has_channel(channel)

  def __repr__(self):
    return self.name


class Task(object):
  def __init__(self, device='Dev1', task_name='', **kwargs):
      self.device = None
      self.task_handle = TaskHandle(0)

      if isinstance(device, Device):
          self.device = device
      else:
          self.device = Device(device)

      CHK(nidaq.DAQmxCreateTask(task_name, ctypes.byref(self.task_handle)))

  ## Some properties

  name = property(fget =
      lambda(x): get_value(x.task_handle,nidaq.DAQmxGetTaskName, str),
      doc = 'Task name')
  channels = property(fget =
      lambda(x): get_value(x.task_handle,nidaq.DAQmxGetTaskChannels, list),
      doc = 'List of all virtual channels in this task')
  number_of_channels = property(fget =
      lambda(x): get_value(x.task_handle,nidaq.DAQmxGetTaskNumChans, uint32),
      doc = 'Number of channels in this task')
  samples_per_channel = property(fget =
      lambda(x): get_value(x.task_handle, nidaq.DAQmxGetSampQuantSampPerChan,
        uint64), fset = lambda(x,y): set_value(x.task_handle,
          nidaq.DAQmxSetSampQuantSampPerChan, uint64(y)),
        doc = 'Number of samples per channel')
  sample_rate = property(fget =
      lambda(x): get_value(x.task_handle, nidaq.DAQmxGetSampClkRate, float64),
      fset = lambda(x,y): set_value(x.task_handle, nidaq.DAQmxSetSampClkRate,
        float64(y)), doc = 'Number of samples per channel')

  def configure_sample_clock_timing(self,sample_rate=10000,
      source='',sample_mode=DAQmx_Val_FiniteSamps,samples_per_channel=100000):
      if type(sample_mode) == str:
          sample_mode = sample_mode_names[sample_mode]

      CHK(nidaq.DAQmxCfgSampClkTiming(self.task_handle,
                                      source,
                                      float64(sample_rate),
                                      DAQmx_Val_Rising,
                                      sample_mode,
                                      uint64(samples_per_channel)))

  def configure_implicit_timing(self, sample_mode, samples_per_channel):
      if type(sample_mode) == str:
          sample_mode = sample_mode_names[sample_mode]
            
      CHK(nidaq.DAQmxCfgImplicitTiming(self.task_handle,
                                       int32(sample_mode),
                                       uint64(samples_per_channel)))


  def wait_until_done(self, timeout=None):
    '''Wait until task is finished.'''
    if timeout is None:
      timeout = -1.
    CHK(nidaq.DAQmxWaitUntilTaskDone(self.task_handle, float64(timeout)))

  def start(self):
    """Start the task."""
    if self.task_handle.value != 0:
      CHK(nidaq.DAQmxStartTask(self.task_handle))

  def stop(self):
    """Stop the task."""
    if self.task_handle.value != 0:
      CHK(nidaq.DAQmxStopTask(self.task_handle))

  def clear(self):
    if self.task_handle.value != 0:
      self.stop()
      CHK(nidaq.DAQmxClearTask(self.task_handle))
    self.task_handle.value = 0

  def __del__(self):
    self.clear()

  def __enter__(self):
    return self

  def __exit__(self, type, value, traceback):
    self.clear()

  def add_analog_voltage_channel(self, **kwargs):
    pass

  def sync_to_ao_trigger(self, pfi_channel=0):
    '''Syncs task to the ao/StartTrigger terminal of the parent
    device.  The pfi_channel parameter sets the internal PFI channel
    to use for signal routing [0].'''

    pfi = '/%s/PFI%d' % (self.device.name, pfi_channel)
    trigger = '/%s/ao/StartTrigger' % self.device.name
        
    connect_terminals(trigger, pfi)
    self.set_digital_trigger(pfi)
        
  def set_digital_trigger(self, terminal, rising=True):
    '''Sets a digital trigger on the specified terminal.'''
    CHK(nidaq.DAQmxCfgDigEdgeStartTrig(self.task_handle,
                                       terminal,
                                       DAQmx_Val_Rising if rising else DAQmx_Val_Falling))

  def create_read_buffer(self, dtype='float64'):
    return create_contiguous_buffer(self.number_of_channels * self.samples_per_channel,dtype=dtype)

class InputTask(Task):
  def __init__(self, device='Dev1', task_name='', **kwargs):
      Task.__init__(self, device, task_name, **kwargs)
      self.read = None

  def add_analog_voltage_channel(self, channel, channel_name='',
                                 terminal_config = DAQmx_Val_Cfg_Default,
                                 min_voltage=-10.0,
                                 max_voltage=10.0):
      '''Add an analog voltage channel to an input task.'''
      channel = norm_channel(channel, self.device, 'analog', 'input')
      if type(terminal_config) == str:
          terminal_config = terminal_config_names[terminal_config]

      CHK(nidaq.DAQmxCreateAIVoltageChan(self.task_handle,
                                         channel, channel_name,
                                         int32(terminal_config),
                                         float64(min_voltage),
                                         float64(max_voltage),
                                         DAQmx_Val_Volts,
                                         None))
      self.read = self.read_analog_float64
      return (self.number_of_channels - 1)

  def add_ci_count_edges_channel(self, channel, channel_name = '',
                                 edge = DAQmx_Val_Rising, initial_count = 0,
                                 count_direction = DAQmx_Val_CountUp):
      channel = norm_channel(channel, self.device, 'counter', 'input')
      CHK(nidaq.DAQmxCreateCICountEdgesChan(self.task_handle, channel, channel_name,
                                            int32(edge), uint32(initial_count),
                                            int32(count_direction)))
      self.read = self.read_counter_scalar_u32
      return (self.number_of_channels - 1)

  def get_analog_minimum(self, channel):
      channel = self._canonical_channel_name(channel, 'analog')
      read = float64()
      CHK(nidaq.DAQmxGetAIMin(self.task_handle, channel, ctypes.byref(read)))
      return read.value

  def set_analog_minimum(self, channel, value):
      channel = self._canonical_channel_name(channel, 'analog')
      CHK(nidaq.DAQmxSetAIMin(self.task_handle, channel, float64(value)))

  def get_analog_maximum(self, channel):
      channel = self._canonical_channel_name(channel, 'analog')
      read = float64()
      CHK(nidaq.DAQmxGetAIMax(self.task_handle, channel, ctypes.byref(read)))
      return read.value

  def set_analog_maximum(self, channel, value):
      channel = self._canonical_channel_name(channel, 'analog')
      CHK(nidaq.DAQmxSetAIMax(self.task_handle, channel, float64(value)))

  def read_analog_float64(self):
      '''Read float64's into an array and return it. Each channel has its own
      column.'''
      read = int32()
      data = self.create_read_buffer(dtype='float64')
      CHK(nidaq.DAQmxReadAnalogF64(self.task_handle, DAQmx_Val_Cfg_Default,
        float64(10.0), DAQmx_Val_GroupByChannel, data.ctypes.data,
        int32(data.size), ctypes.byref(read), None))
      if read.value != self.samples_per_channel:
        warn('Read %d samples per channel of %d.' % (read.value,
          self.samples_per_channel))
      if self.number_of_channels > 1:
        a = data[:read.value * self.number_of_channels]
        return a.reshape((self.number_of_channels,self.samples_per_channel)).transpose()
      else:
        return data[:read.value * self.number_of_channels]

  def read_analog_scalar_float64(self, timeout=10.0):
      read = float64()
      CHK(nidaq.DAQmxReadAnalogScalarF64(self.task_handle,
                                         float64(timeout),
                                         ctypes.byref(read),
                                         None))
      return read.value

  def read_counter_scalar_u32(self, timeout = 10.0):
      value = int32()
      CHK(nidaq.DAQmxReadCounterScalarU32(self.task_handle, float64(timeout),
                                          ctypes.byref(value), None))
      return value.value

  def read_counter_u32(self, timeout = 10.0):
      read = int32()
      data = self.create_read_buffer(dtype=uint32)
      CHK(nidaq.DAQmxReadCounterU32(self.task_handle,
                                    int32(self.samples_per_channel),
                                    float64(timeout),
                                    data.ctypes.data,
                                    int32(data.size),
                                    ctypes.byref(read), None))
      if read.value != self.samples_per_channel:
        warn('Read %d samples per channel of %d.' % (read.value,
          self.samples_per_channel))
      if self.number_of_channels > 1:
        a = data[:read.value * self.number_of_channels]
        return a.reshape((self.number_of_channels,self.samples_per_channel)).transpose()
      else:
        return data[:read.value * self.number_of_channels]
      

  def _canonical_channel_name(self, channel, type):
      return Task._canonical_channel_name(self, channel, type, 'input')




class OutputTask(Task):
    def __init__(self, device='Dev1', task_name='', **kwargs):
        Task.__init__(self, device, task_name, **kwargs)

    def add_analog_voltage_channel(self, channel, channel_name='',
                                   min_voltage=-10.0, max_voltage=10.0):
        '''Add an analog voltage channel to an output task.'''
        channel = norm_channel(channel, self.device, 'analog', 'output')
        CHK(nidaq.DAQmxCreateAOVoltageChan(self.task_handle,
                                           channel,
                                           channel_name,
                                           float64(min_voltage),
                                           float64(max_voltage),
                                           DAQmx_Val_Volts, None))
        return (self.number_of_channels - 1)

    def add_co_pulse_channel_freq(self, channel, channel_name='',
                                  units=DAQmx_Val_Hz,
                                  idle_state=DAQmx_Val_Low,
                                  initial_delay=0,
                                  frequency=500, duty_cycle=0.5):
        channel = norm_channel(channel, self.device, 'counter', 'output')
        CHK(nidaq.DAQmxCreateCOPulseChanFreq(self.task_handle,
                                             channel,
                                             channel_name,
                                             int32(units),
                                             int32(idle_state),
                                             float64(initial_delay),
                                             float64(frequency),
                                             float64(duty_cycle)))
        return (self.number_of_channels - 1)

    def write_analog_float64(self, data, auto_start=False, timeout=10.0):
        '''Write float64's into an array and return it. TODO: handle
    multiple channels in an output task.'''
        written = uint32(0)

        buffer = create_contiguous_buffer(len(data))
        buffer[:] = data

        CHK(nidaq.DAQmxWriteAnalogF64(
                self.task_handle,
                int32(len(data)),
                bool32(auto_start),
                float64(timeout),
                DAQmx_Val_GroupByChannel,
                buffer.ctypes.data,
                ctypes.byref(written),
                None))

        written = written.value
        
        if written != len(data):
            raise RuntimeError("Attempt to write analog data only wrote %d/%d samples!" % (written, len(data)))

        return written

    def write_analog_scalar_float64(self, value, auto_start=False,
                                    timeout=10.0):
        CHK(nidaq.DAQmxWriteAnalogScalarF64(
                self.task_handle, bool32(auto_start),
                float64(timeout), float64(value), None))

    def write(self, value, auto_start=False, timeout=10.0):
        if type(value) == float:
            self.write_analog_scalar_float64(value, auto_start, timeout)
        elif type(value) == numpy.ndarray:
            self.write_analog_float64(value, auto_start, timeout)
        else:
            raise ValueError('Unable to dispatch write function on the type of value.')

    def _canonical_channel_name(self, channel, type):
        return Task._canonical_channel_name(self, channel, type, 'output')

  
def test_sync():
    dev = Device('Dev1')
    print dev.analog_input_voltage_ranges

    readtask = InputTask(dev)
    readtask.add_analog_voltage_channel('ai2', terminal_config = 'rse')
    readtask.add_analog_voltage_channel('ai3', terminal_config = 'rse')

    readtask.configure_sample_clock_timing(sample_rate=10000, samples_per_channel=50000)

    writetask = OutputTask(dev)
    writetask.add_analog_voltage_channel('ao0')

    writetask.configure_sample_clock_timing(sample_rate=10000, samples_per_channel=50000)

    m = numpy.linspace(-5.0,5.0, 50000)
    writetask.write(m)

    readtask.sync_to_ao_trigger()
    print readtask.channels
    print readtask.number_of_channels
    print readtask.samples_per_channel
    print writetask.channels
    readtask.start()
    writetask.start()
    readtask.wait_until_done()
    
    n = readtask.read()
    print n.mean(axis=0)
    print n.std(axis=0)
    
    writetask.stop()
    readtask.stop()
    
    del(readtask)
    del(writetask)
    
    result = numpy.column_stack((m,n))

def test_write(channel='ao0',num=5.0, sample_rate=None):
    dev = Device('Dev1')
    writetask = OutputTask(dev)
    writetask.add_analog_voltage_channel(channel)
    if type(num) == int or type(num) == float:
        writetask.write(float(num), True)
    elif sample_rate is not None:
        writetask.configure_sample_clock_timing(sample_rate=sample_rate,
                                                samples_per_channel=len(num))
        writetask.write(num,True)
        writetask.wait_until_done()
        writetask.stop()
        writetask.clear()
    else:
        print 'Unknown'
        
def test_read(channel='ai3'):
    dev = Device('Dev1')
    task = InputTask(dev)
    task.add_analog_voltage_channel(channel,terminal_config='rse')
    return task.read_analog_scalar_float64()

def test_ramp_task(channel='ao0', Vmin=0.0, Vmax=5.0, num_samples=1000, sample_rate=10000):
    from numpy import linspace, hstack
    dev = Device('Dev1')
    writetask = OutputTask(dev)
    writetask.add_analog_voltage_channel(channel)
    if num_samples % 2 == 1:
      num_samples += 1
    writetask.configure_sample_clock_timing(sample_rate=sample_rate,
                                            samples_per_channel=num_samples)
    up = linspace(Vmin, Vmax, num_samples/2)
    down = linspace(Vmax, Vmin, num_samples/2)
    form = hstack((up,down))
    writetask.write(form)
    return writetask

def test_count_task(channel='Dev1/ctr0', integration_time=1.0, N = 2):
    from time import sleep
    with InputTask(Device('Dev1')) as t:
        t.add_ci_count_edges_channel(channel)
        for i in range(N):
            t.start()
            sleep(integration_time)
            print '%d counts' % t.read()
            t.stop()

def test_pulse_task(channel='Dev1/ctr1', frequency=500,duty_cycle=0.5):
    with OutputTask(Device('Dev1')) as t:
        t.add_co_pulse_channel_freq(channel, frequency=frequency,
                                    duty_cycle=duty_cycle)
        t.configure_implicit_timing('finite', 10000)
        t.start()
        t.wait_until_done()
        t.stop()

