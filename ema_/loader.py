import numpy as np
import struct
import time
import boto3
import json
import os
import pandas as pd
from os import listdir
from os.path import isfile, join


def dw_files(save_path, file_numbers, credentials_path):
    with open(credentials_path) as f:
        cred = json.load(f)

    s3_client = boto3.client('s3',
                             endpoint_url='https://cloud-areapd.pd.infn.it:5210',
                             aws_access_key_id=cred["access_key"],
                             aws_secret_access_key=cred["secret_key"],
                             verify=False)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i in file_numbers:
        filename = f'data_{i:06}.dat'
        save_name = save_path + "/" + filename
        s3_client.download_file('mapd-minidt-batch', filename, save_name)
        print(f"Ho scaricato il file {save_name}")


def timeit(func, count, args=None, kwargs=None):
    args = args or []
    kwargs = kwargs or {}

    total = 0
    for i in range(count):
        t0 = time.time()
        func(*args, **kwargs)
        t1 = time.time()
        total += t1 - t0
    print(f"AVG: {total / count:.2f} s")


def vanilla_loading(path, output=True):
    hits = 0
    non_hits = 0
    invalid_hits = 0
    valid_chan = 0
    invalid_chan = 0
    valid_fpga = 0
    invalid_fpga = 0
    valid_tdc = 0
    invalid_tdc = 0
    with open(path, "rb") as f:
        while True:
            bdata = f.read(BYTES_PER_READ)
            # Questa riga dice come spachettare quel paccone di byte appena letto
            # < serve per dire che i byte sono salvati come little endian
            # {len(bdata) // 8} specifica il numero di quanti "64 bit" ci sono nei bytes letti
            #  ce ne aspetteremo "BYTES_PER_READ // 8" ma se il numero di hits non è divisibile per
            #  HITS_PER_READ allora ne potrebbe leggere di meno
            # Q specifica un unsigned long long (8 bytes), andava bene anche un q (signed long long)
            #  tanto lavoriamo bit a bit, non vogliamo tradurre direttamente i valori letti in interi
            unpack_str = f"<{len(bdata) // 8}Q"
            if not bdata:
                break

            for data in struct.unpack(unpack_str, bdata):
                # solita tecnica di shiftare i bit in modo da averceli in fondo e dopo fare l'AND
                tdc = data & 0b11111
                bx = (data >> 5) & 0b1111_11111111
                orbit = (data >> 17) & 0b11111111_11111111_11111111_11111111
                chan = (data >> 49) & 0b1_11111111
                fpga = (data >> 58) & 0b111
                head = data >> 61

                # Piccoli controlli di sensatezza dei dati
                if head == 2:
                    hits += 1
                elif 0 <= head <= 5:
                    non_hits += 1
                else:
                    invalid_hits += 1

                if 0 <= chan <= 128:  # we expect 0-127 for the real channels and 128 for the t0
                    valid_chan += 1
                else:
                    invalid_chan += 1

                if 0 <= fpga <= 1:  # we expect 0 or 1
                    valid_fpga += 1
                else:
                    invalid_fpga += 1

                if 0 <= tdc <= 30:  # I think this is not a proper way to check if it's correct
                    valid_tdc += 1
                else:
                    invalid_tdc += 1

        if output:
            print("Valid hits:", hits, "Valid non hits:", non_hits, "Invalid hits:", invalid_hits)
            print("Valid channels:", valid_chan, "Invalid channels:", invalid_chan)
            print("Valid FPGA:", valid_fpga, "Invalid FPGA:", invalid_fpga)
            print("Valid TDC:", valid_tdc, "Invalid TDC:", invalid_tdc)

        raise NotImplementedError()


def numpy_loading(path, output=True, analyze=True):
    with open(path, "rb") as f:
        # se le dimensioni del file iniziano essere vicino ai giga potrebbe non essere buono
        bdata = f.read()

    if (len(bdata) % 8) != 0:
        print("iL NUMERO DI BYTE NON E' MULTIPLO DI 8")
        exit(1)

    # NON E' STATO SPECIFICATO L'ENDIANESS DEL FILE (prende quello della macchina, potrebbe non andare sempre bene)
    data = np.frombuffer(bdata, dtype=np.ulonglong, count=len(bdata) // 8)
    tdcs = data & 0b11111
    bxs = (data >> 5) & 0b1111_11111111
    orbits = (data >> 17) & 0b11111111_11111111_11111111_11111111
    chans = (data >> 49) & 0b1_11111111
    fpgas = (data >> 58) & 0b111
    heads = data >> 61

    if analyze:
        hits = np.sum(heads == 2)
        non_hits = np.sum((heads >= 0) & (heads <= 5)) - hits
        invalid_hits = np.sum((heads < 0) | (heads > 5))

        valid_chan = np.sum((chans >= 0) & (chans <= 128))  # we expect 0-127 for the real channels and 128 for the t0
        invalid_chan = np.sum((chans < 0) | (chans > 128))

        valid_fpga = np.sum((fpgas == 0) | (fpgas == 1))
        invalid_fpga = np.sum((fpgas != 0) & (fpgas != 1))

        # I think this is not a proper way to check if it's correct
        valid_tdc = np.sum((tdcs >= 0) & (tdcs <= 30))
        invalid_tdc = np.sum((tdcs < 0) | (tdcs > 30))

        if output:
            print("Valid hits:", hits, "Valid non hits:", non_hits, "Invalid hits:", invalid_hits)
            print("Valid channels:", valid_chan, "Invalid channels:", invalid_chan)
            print("Valid FPGA:", valid_fpga, "Invalid FPGA:", invalid_fpga)
            print("Valid TDC:", valid_tdc, "Invalid TDC:", invalid_tdc)

        if invalid_hits:
            raise ValueError(f"There are {invalid_hits} invalid hits with 0 < head or head > 5")

    mat = np.column_stack((tdcs, bxs, orbits, chans, fpgas, heads))
    mat = mat[(mat[:, 5] == 2)]
    # mat = np.array(mat, dtype=[("TDC", np.uint8), ("BX", np.uint16), ("ORBIT", np.uint32),
    #                            ("CHAN", np.uint16), ("FPGA", np.uint8), ("HEAD", np.uint8)])
    # assicurarsi che questo passaggio non faccia modifiche, se non ridurre da un inutile 64 bit a un sufficiente 32 bit
    # pd.DataFrame(data=mat, columns=["TDC", "BX", "ORBIT", "CHAN", "FPGA", "HEAD"])

    return mat.astype(np.uint32)


HITS_PER_READ = 10000
BYTES_PER_READ = HITS_PER_READ * 8

if __name__ == "__main__":
    pass

    # PER SCARICAREEEEEEEEEEEEEEEEEEEEE I FILES
    # dw_files(save_path="./dataset", file_numbers=np.arange(0, 81), credentials_path="./credentials.json")

    # PER LEGGERE UN FILE:
    # filename = "./dataset/data_000000.dat"
    # mat = numpy_loading(filename, output=False, analyze=False)

    # PER LEGGERE TUTTI I FILES
    # onlyfiles = [f for f in listdir("./dataset") if isfile(join("./dataset", f))]
    # onlyfiles = ["./dataset/" + f for f in onlyfiles if f.endswith(".dat")]
    #
    # time0 = time.time()
    # for filename in onlyfiles:
    #     print("Loading", filename)
    #     mat = numpy_loading(filename, output=False, analyze=False)
    # time1 = time.time()
    # print(f"Caricati {len(onlyfiles)} files in {time1 - time0:.3g} secondi")
