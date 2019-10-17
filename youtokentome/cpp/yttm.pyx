from cython.operator cimport dereference as deref
from cpython.buffer cimport PyObject_GetBuffer, PyBuffer_Release, \
    PyBUF_READ, PyBUF_SIMPLE, PyBUF_ANY_CONTIGUOUS
from libc.string cimport memcpy
from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool
import os
from pathlib import Path


cdef extern from "utils.h" namespace "srcd":
    ctypedef void (*py_write_func)(void *self, const char *buffer, int size)
    ctypedef void (*py_read_func)(void *self, char *buffer, int size)
    ctypedef string (*py_name_func)(void *self)

    cdef cppclass StreamReader:
        @staticmethod
        unique_ptr[StreamReader] open(const string &file_name_)

        @staticmethod
        unique_ptr[StreamReader] assemble(py_read_func read, py_name_func name, void *self)

    cdef cppclass StreamWriter:
        @staticmethod
        unique_ptr[StreamWriter] open(const string &file_name_)

        @staticmethod
        unique_ptr[StreamWriter] assemble(py_write_func write, py_name_func name, void *self)


cdef extern from "bpe.h" namespace "srcd":
    cdef cppclass SpecialTokens:
        int pad_id
        int unk_id
        int bos_id
        int eos_id

    cdef cppclass BpeConfig:
        double character_coverage
        int n_threads
        SpecialTokens special_tokens

    cdef cppclass Status:
        int code
        string message

    cdef cppclass BPEState:
        void dump(StreamWriter &fout)


cdef extern from "bpe.h" namespace "srcd":
    Status train_bpe(StreamReader &input, StreamWriter &output, int vocab_size,
                     const BpeConfig& bpe_config)


cdef extern from "bpe.h" namespace "srcd":
    cdef cppclass BaseEncoder:
        BPEState bpe_state
        BaseEncoder(StreamReader& model_path, int n_threads, Status* status)

        Status encode_as_ids(const vector[string] &sentences, vector[vector[int]]* ids, bool bos, bool eos, bool reverse) const
        Status encode_as_subwords(const vector[string]& sentences, vector[vector[string]]* subwords, bool bos, bool eos, bool reverse) const

        Status encode_cli(string output_type, bool stream, bool bos, bool eos, bool reverse) const

        Status decode_cli() const

        void vocab_cli(bool verbose) const

        Status id_to_subword(int id, string* subword) const

        int subword_to_id(const string &subword) const
        Status decode(const vector[vector[int]]& ids, vector[string]* output) const
        int vocab_size() const
        vector[string] vocabulary() const


cdef extern from "Python.h":
    object PyMemoryView_FromMemory(char *mem, ssize_t size, int flags)


cdef void write_callback(void *self, const char *buffer, int size):
    (<object>self).write(PyMemoryView_FromMemory(<char *>buffer, size, PyBUF_READ))


cdef void read_callback(void *self, char *buffer, int size):
    pybuf = (<object>self).read(size)
    cdef Py_buffer bufmem
    PyObject_GetBuffer(pybuf, &bufmem, PyBUF_SIMPLE | PyBUF_ANY_CONTIGUOUS)
    memcpy(buffer, <char *>bufmem.buf, size)
    PyBuffer_Release(&bufmem)


cdef string name_callback(void *self):
    return (<object>self).name


cdef class BPE:
    cdef BaseEncoder* encoder

    def __dealloc__(self):
        del self.encoder

    def __init__(self, fobj, n_threads=-1):
        cdef Status status
        cdef unique_ptr[StreamReader] reader = StreamReader.assemble(
            read_callback, name_callback, <void*>fobj)
        self.encoder = new BaseEncoder(deref(reader), n_threads, &status)
        if status.code != 0:
            raise ValueError(status.message.decode())

    @staticmethod
    def train(data,
              fobj,
              vocab_size,
              coverage=1.0,
              n_threads=-1,
              pad_id=0,
              unk_id=1,
              bos_id=2,
              eos_id=3):

        cdef BpeConfig bpe_config
        bpe_config.character_coverage = coverage
        bpe_config.n_threads = n_threads
        bpe_config.special_tokens.pad_id = pad_id
        bpe_config.special_tokens.unk_id = unk_id
        bpe_config.special_tokens.bos_id = bos_id
        bpe_config.special_tokens.eos_id = eos_id
        cdef unique_ptr[StreamReader] reader = StreamReader.assemble(
            read_callback, name_callback, <void*>data)
        cdef unique_ptr[StreamWriter] writer = StreamWriter.assemble(
            write_callback, name_callback, <void*>fobj)

        cdef Status status = train_bpe(deref(reader), deref(writer), vocab_size, bpe_config)
        if status.code != 0:
            raise ValueError(status.message.decode())

    def encode(self, sentences, output_type, bos, eos, reverse):
        cdef vector[string] s
        cdef vector[vector[string]] ret_subwords
        cdef vector[vector[int]] ret_ids
        cdef Status status
        if output_type == 'id':
            if isinstance(sentences, str):
                s = [sentences.encode()]
                assert len(s) == 1
                status = self.encoder.encode_as_ids(s, &ret_ids, bos, eos, reverse)
                if status.code != 0:
                    raise ValueError(status.message.decode())
                return ret_ids[0]

            assert isinstance(sentences, list) or isinstance(sentences, tuple)
            s = [x.encode() for x in sentences]
            status = self.encoder.encode_as_ids(s, &ret_ids, bos, eos, reverse)
            if status.code != 0:
                raise ValueError(status.message.decode())
            return ret_ids
        elif output_type == 'subword':
            if isinstance(sentences, str):
                s = [sentences.encode()]
                status = self.encoder.encode_as_subwords(s, &ret_subwords, bos, eos, reverse)
                if status.code != 0:
                    raise ValueError(status.message.decode())
                assert len(ret_subwords) == 1
                return [piece.decode() for piece in ret_subwords[0]]

            assert isinstance(sentences, list) or isinstance(sentences, tuple)
            s = [x.encode() for x in sentences]
            status = self.encoder.encode_as_subwords(s, &ret_subwords, bos, eos, reverse)
            if status.code != 0:
                raise ValueError(status.message.decode())
            return [[piece.decode() for piece in sentence] for sentence in ret_subwords]
        else:
            raise ValueError('output_type must be equal to "id" or "subword"')

    def save(self, fobj):
        cdef unique_ptr[StreamWriter] writer = StreamWriter.assemble(
            write_callback, name_callback, <void*>fobj)
        self.encoder.bpe_state.dump(deref(writer));

    def subword_to_id(self, subword):
        return self.encoder.subword_to_id(subword.encode())

    def id_to_subword(self, id):
        cdef string subword
        cdef Status status = self.encoder.id_to_subword(id, &subword)
        if status.code != 0:
            raise ValueError(status.message.decode())
        return subword.decode()

    def decode(self, ids):
        assert isinstance(ids, list)
        if len(ids) > 0 and isinstance(ids[0], int):
            ids = [ids]
        cdef vector[string] sentences
        cdef Status status = self.encoder.decode(ids, &sentences)
        if status.code != 0:
            raise ValueError(status.message.decode())
        return [sentence.decode() for sentence in sentences]

    def vocab_size(self):
        return self.encoder.vocab_size();

    def vocab(self):
        cdef vector[string] vocab = self.encoder.vocabulary()
        return [token.decode() for token in vocab]

    def encode_cli(self, output_type, stream, bos, eos, reverse):
        cdef Status status = self.encoder.encode_cli(output_type.encode(), stream, bos, eos, reverse)
        if status.code != 0:
            raise ValueError(status.message.decode())

    def decode_cli(self):
        cdef Status status = self.encoder.decode_cli()
        if status.code != 0:
            raise ValueError(status.message.decode())

    def vocab_cli(self, verbose):
        self.encoder.vocab_cli(verbose)

