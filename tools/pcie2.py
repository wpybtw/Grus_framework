#!python3
import socket 
import time
# from pynvml import *
from py3nvml.py3nvml import *
nvmlInit()
strResult=''
gpu=0 #GPU 0
handle = nvmlDeviceGetHandleByIndex(gpu)
print( 'GPU name ' + str(nvmlDeviceGetName(handle)))
print( 'Driver Version ' + str(nvmlSystemGetDriverVersion()))
print( 'VBOIS Version '+str(nvmlDeviceGetVbiosVersion(handle)))
util = nvmlDeviceGetUtilizationRates(handle)
print( 'GPU Utilization ' + str(util.gpu) + ' %')
print( 'GPU Memory Utilization ' +  str(util.memory) + ' %')
#GPU PCIe Throughput
while True:
    time.sleep(0.1)
    tx_mb = nvmlDeviceGetPcieThroughput(handle, NVML_PCIE_UTIL_TX_BYTES)/1024
    rx_mb = nvmlDeviceGetPcieThroughput(handle, NVML_PCIE_UTIL_RX_BYTES)/1024
    print('PCIe tx ' + str(round(tx_mb)) + ' MB/s' +'        rx ' + str(round(rx_mb)) + ' MB/s')

nvmlShutdown()