"""Diagnostic packet reader: PortAudio opens/starts the stream; WASAPI alone reads it.

No vtable patching or DLL replacement. COM interface is marshalled to the reader
thread; GetBuffer and ReleaseBuffer are paired there. Exclusive PCM16 stereo only.
"""
import ctypes as c
from collections import deque
import uuid


def guid(value):
    return (c.c_ubyte * 16).from_buffer_copy(uuid.UUID(value).bytes_le)


class PacketTrace:
    def __init__(self):
        self.rows = deque(maxlen=256)
        self.total = self.discontinuities = self.timestamp_errors = 0
        self.trigger = None
        self.tail = 64
        self.frozen = False
        self.expected_position = None
        self.identity_offset = None

    def add(self, row):
        self.total += 1
        self.discontinuities += bool(row['flags'] & 1)
        self.timestamp_errors += bool(row['flags'] & 4)
        row['sequence'] = self.total
        row['position_gap'] = None if self.expected_position is None else row['device_position'] - self.expected_position
        self.expected_position = row['device_position'] + row['frames']
        first, last = row.get('first_word', 0), row.get('last_word', 0)
        encoded = first & 0xf0000000 == 0xa0000000
        offset = (first & 0x0fffffff) - row['device_position'] if encoded else None
        row['source_offset'] = offset
        row['source_discontinuity'] = bool(encoded and (
            (self.identity_offset is not None and offset != self.identity_offset)
            or (last & 0xf0000000 == 0xa0000000 and (last & 0x0fffffff) - (first & 0x0fffffff) != row['frames'] - 1)))
        if encoded:
            self.identity_offset = offset
        if self.frozen:
            return
        self.rows.append(row)
        # Initial discontinuity is common at startup; preserve it without freezing.
        anomaly = self.total > 1 and (row['position_gap'] != 0 or row['flags'] & 5 or row['source_discontinuity'])
        if anomaly and self.trigger is None:
            self.trigger = self.total
        elif self.trigger is not None:
            self.tail -= 1
            self.frozen = self.tail == 0

    def report(self):
        return dict(total_packets=self.total, discontinuity_packets=self.discontinuities,
                    timestamp_error_packets=self.timestamp_errors, trigger=self.trigger,
                    frozen=self.frozen, records=list(self.rows))


class WasapiCapture:
    CLIENT = '1CB9AD4C-DBFA-4c32-B178-C2F568A703B2'
    CAPTURE = 'C8ADBD64-E71E-48a0-A4DE-185C395CD317'

    def __init__(self, stream, sd):
        self.trace = PacketTrace()
        self.ole = c.OleDLL('ole32')
        self.ole.CoUninitialize.restype = None
        self.client = c.c_void_p()
        self.capture = c.c_void_p()
        self.marshalled = c.c_void_p()
        self.initialized = False
        library = c.CDLL(sd._libname)
        get_client = library.PaWasapi_GetAudioClient
        get_client.argtypes = [c.c_void_p, c.POINTER(c.c_void_p), c.c_int]
        get_client.restype = c.c_int
        borrowed = c.c_void_p()
        error = get_client(int(sd._ffi.cast('uintptr_t', stream._ptr)), c.byref(borrowed), 0)
        if error:
            raise RuntimeError(f'PaWasapi_GetAudioClient failed: {error}')
        marshal = self.ole.CoMarshalInterThreadInterfaceInStream
        marshal.argtypes = [c.c_void_p, c.c_void_p, c.POINTER(c.c_void_p)]
        marshal(guid(self.CLIENT), borrowed, c.byref(self.marshalled))

    @staticmethod
    def method(pointer, index, result, *arguments):
        table = c.cast(pointer, c.POINTER(c.POINTER(c.c_void_p))).contents
        return c.WINFUNCTYPE(result, c.c_void_p, *arguments)(table[index])

    @staticmethod
    def check(hr, operation):
        if hr < 0:
            raise RuntimeError(f'{operation}: HRESULT 0x{hr & 0xffffffff:08x}')

    def open(self):
        self.ole.CoInitializeEx(None, 0)
        self.initialized = True
        unmarshal = self.ole.CoGetInterfaceAndReleaseStream
        unmarshal.argtypes = [c.c_void_p, c.c_void_p, c.POINTER(c.c_void_p)]
        marshalled = self.marshalled
        self.marshalled = c.c_void_p()
        unmarshal(marshalled, guid(self.CLIENT), c.byref(self.client))
        service = self.method(self.client, 14, c.c_long, c.c_void_p, c.POINTER(c.c_void_p))
        self.check(service(self.client, guid(self.CAPTURE), c.byref(self.capture)), 'GetService')
        self.padding = self.method(self.client, 6, c.c_long, c.POINTER(c.c_uint32))
        self.get_buffer = self.method(self.capture, 3, c.c_long, c.POINTER(c.c_void_p),
                                      c.POINTER(c.c_uint32), c.POINTER(c.c_uint32),
                                      c.POINTER(c.c_uint64), c.POINTER(c.c_uint64))
        self.release_buffer = self.method(self.capture, 4, c.c_long, c.c_uint32)

    def read(self, qpc_now):
        available = c.c_uint32()
        self.check(self.padding(self.client, c.byref(available)), 'GetCurrentPadding')
        if not available.value:
            return None
        pointer = c.c_void_p()
        frames, flags = c.c_uint32(), c.c_uint32()
        position, timestamp = c.c_uint64(), c.c_uint64()
        before = qpc_now()
        hr = self.get_buffer(self.capture, c.byref(pointer), c.byref(frames), c.byref(flags),
                             c.byref(position), c.byref(timestamp))
        self.check(hr, 'GetBuffer')
        if hr == 0x8890001:  # AUDCLNT_S_BUFFER_EMPTY: no matching ReleaseBuffer.
            return None
        try:
            size = frames.value * 4
            data = bytes(size) if flags.value & 2 else c.string_at(pointer, size)
            self.trace.add(dict(qpc=before, padding=available.value, frames=frames.value,
                                flags=flags.value, device_position=position.value,
                                packet_qpc_100ns=timestamp.value,
                                first_word=int.from_bytes(data[:4], 'little'),
                                last_word=int.from_bytes(data[-4:], 'little')))
        finally:
            self.check(self.release_buffer(self.capture, frames.value), 'ReleaseBuffer')
        return data

    def close(self):
        for pointer in (self.capture, self.client):
            if pointer.value:
                self.method(pointer, 2, c.c_ulong)(pointer)
                pointer.value = None
        if self.initialized:
            self.ole.CoUninitialize()
            self.initialized = False

    def close_pending(self):
        # Main-thread cleanup if opening/starting failed before unmarshalling.
        if self.marshalled.value:
            release_data = self.ole.CoReleaseMarshalData
            release_data.argtypes = [c.c_void_p]
            try:
                release_data(self.marshalled)
            finally:
                self.method(self.marshalled, 2, c.c_ulong)(self.marshalled)
                self.marshalled.value = None
