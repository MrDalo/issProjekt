import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import soundfile as sf


def dft(data, fs):
    ### pri selectedFrame som si vybral 
    selectedFrame = data[26*512:(27*512)+512]

    N = selectedFrame.size
    n = np.arange(N)
    k = n.reshape((N,1))
    e = np.exp(-2j* np.pi * k * n / N)

    DftArray = np.dot(e, selectedFrame)
    return DftArray


def questOne():
    #nacitanie signalu
    data, fs = sf.read('xkrali20.wav')

    ### Uloha 1.
    print("delka ve vzorcich: ",data.size) 
    print("delka v sekundach: ", data.size/fs)
    print("minimalna hodnota: ", data.min())
    print("maximalna hodnota: ", data.max())


    #vyplotenie grafu
    time = np.arange(data.size)/fs
    plt.figure(figsize=(10, 5))
    plt.plot(time, data)

    plt.gca().set_xlabel('$t[s]$')
    plt.gca().set_title('Zvukový signál')
    plt.gca().set_ylabel('$f\ \' $')

    plt.tight_layout()

    plt.show()

def questTwo():
    ### Uloha 2.
    fs, data = wavfile.read('xkrali20.wav')

    #ustrednenie
    data = data - np.mean(data)


    #normalizacia input signalu
    data = data / 2**15
    matrixOfData = np.zeros(shape=(round(data.size/512)-2, 1024))



    for i in range(round(data.size/512)-2):
        arrayOfData = data[i*512:((i+1)*512)+512]
        matrixOfData[i]=arrayOfData
        if i == 26:
            time = np.arange(arrayOfData.size)/fs
            plt.figure(figsize=(10, 5))
            plt.plot(time, arrayOfData)
            plt.gca().set_title(f"Predspracovanie signalu, frame: {i}")
            plt.show()
            

  
    ### vyskreslenie normalizovaneho a ustredneneho signalu
    #print(matrixOfData)
    #plt.figure(figsize=(10, 5))
    #time = np.arange(data.size)/fs
    #plt.plot(time, data)
    #plt.show()

    ### Uloha 3.

    DftArray = dft(data, fs)    
    plt.figure(figsize=(10, 5))
    time = np.arange(512)*(fs//2)/512   
    plt.plot(time, abs(DftArray[:512]))
    plt.gca().set_title("DFT")
    plt.show()


    #selectedFrame = data[26*512:(27*512)+512]
    #selectedFrame = np.fft.fft(selectedFrame)
    #time = np.arange(512)
    #time = time * fs//2/512
    #plt.figure(figsize=(10, 5))
    #plt.plot(time, abs(selectedFrame[:512]))
    #plt.gca().set_title("DFT")
    #plt.show()





questOne()
questTwo()

