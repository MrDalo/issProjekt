import numpy as np
from scipy.io import wavfile
import scipy.signal as signal
import matplotlib.pyplot as plt
import soundfile as sf


def dft(selectedFrame):

    N = selectedFrame.size
    # vektor, v ktorom su ulozene hodnoty od 0 po N-1
    n = np.arange(N)
    #otocenie vektoru z vektoru 1xN do matice(vektoru) Nx1, pre nasobenie matic(vektorov) vo funkcii np.dot
    k = n.reshape((N,1))
    #vytvorenie pola e^rovnica v zatvorkach funkcie np.exp
    e = np.exp(-2j* np.pi * k * n / N)

    #vynasobenie dvoch matic
    DftArray = np.dot(e, selectedFrame)

    # Princip vypoctu DFT som poznal z prednasok, avsak problem mi tvorila neznalost numpy funkcii a prace s maticami v python numpy
    # ako studijny zdroj som pouzil materialy z online spracovaneho notebooku "Python Programming and Numerical Methods - A Guide for Engineers and Scientists"
    # presny zdroj: https://pythonnumericalmethods.berkeley.edu/notebooks/chapter24.02-Discrete-Fourier-Transform.html
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

            plt.gca().set_xlabel('$t[s]$')
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
    # pri selectedFrame som si vybral frame cislo 26
    selectedFrame = data[26*512:(27*512)+512]
    DftArray = dft(selectedFrame)    
    plt.figure(figsize=(10, 5))
    time = np.arange(512)*(fs//2)/512   
    plt.plot(time, abs(DftArray[:512]))
    plt.gca().set_xlabel('$f[HZ] $')
    plt.gca().set_title("DFT")
    plt.show()


    #selectedFrame = data[26*512:(27*512)+512]
    #selectedFrame = np.fft.fft(selectedFrame)
    ### selectedFrame = np.fft.fft(data)
    #time = np.arange(512)
    #time = time * fs//2/512
    ### time = np.arange(data.size)/fs
    #plt.figure(figsize=(10, 5))
    #plt.plot(time, abs(selectedFrame))
    #plt.gca().set_title("DFT")
    #plt.show()


    ### Uloha 4.
    f, t, sgr = signal.spectrogram(data, fs)
    sgr_log = 10 * np.log10(sgr+1e-20) 
    plt.figure(figsize=(10,5))
    plt.pcolormesh(t,f,sgr_log)
    plt.gca().set_xlabel('t [s]')
    plt.gca().set_ylabel('f [Hz]')
    cbar = plt.colorbar()
    cbar.set_label('Spektralní hustota výkonu [dB]', rotation=270, labelpad=15)

    plt.tight_layout()
    plt.show()


    ### skuska vyriesenia ulohy podla napovedy v zadani
    #matrixOfExp = np.zeros(shape=(round(data.size/512)-2, 1024))
    #for i in range(round(data.size/512)-2):
    #    arrayOfData = data[i*512:((i+1)*512)+512]
    #    matrixOfExp[i]=dft(arrayOfData)
    #    matrixOfExp[i] = 10 * np.log10(matrixOfExp[i]**2) 
    #plt.figure(figsize=(10,5))
    #plt.pcolormesh(t,f,matrixOfExp)
    #plt.gca().set_xlabel('t [s]')
    #plt.gca().set_ylabel('f [Hz]')
    #cbar = plt.colorbar()
    #cbar.set_label('Spektralní hustota výkonu [dB]', rotation=270, labelpad=15)
    #plt.tight_layout()
    #plt.show()








questOne()
questTwo()

