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

    plt.tight_layout()

    plt.show()

def nextQuests():
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
    plt.gca().set_title('Spektrogram')
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


    ### Uloha 6.
    f1 = 972
    f2 = 1924
    f3 = 2910
    f4 = 3885

    N = data.size
    time = np.arange(N)/fs
    y1 = np.cos(2*np.pi*f1 * time)
    y2 = np.cos(2*np.pi*f2 * time)
    y3 = np.cos(2*np.pi*f3 * time)
    y4 = np.cos(2*np.pi*f4 * time)
    Y = y1+y2+y3+y4
    wavfile.write("4cos.wav", fs, Y.astype(np.float32))

    
    #plt.figure(figsize=(10, 5))
    #plt.plot(time, Y)
    #plt.gca().set_xlabel('$t[s]$')
    #plt.gca().set_title('Zvukový signál')
    #plt.gca().set_ylabel('$f\ \' $')
    #plt.tight_layout()
    #plt.show()



    ### Uloha 7.

    fig, ax = plt.subplots(2, 2, figsize=(10,5))

    sampleNumbers = 100
    imp = [1, *np.zeros(data.size-1)]
    b1, a1 = signal.butter(4, [(f1-30)/(fs/2),(f1+30)/(fs/2)], btype='bandstop', analog=False, output='ba')
    h = signal.lfilter(b1, a1, imp[:sampleNumbers])
    ax[0,0].stem(np.arange(sampleNumbers), h, basefmt= ' ')
    ax[0,0].set_title('Impulzna odozva pre f1')
    ax[0,0].set_xlabel('N')
    
    b2, a2 = signal.butter(6, [(f2-30)/(fs/2),(f2+30)/(fs/2)], btype='bandstop', analog=False, output='ba')
    h = signal.lfilter(b2, a2, imp[:100])
    ax[0,1].stem(np.arange(sampleNumbers), h, basefmt= ' ')
    ax[0,1].set_title('Impulzna odozva pre f2')
    ax[0,1].set_xlabel('N')
        
    b3, a3 = signal.butter(4, [(f3-30)/(fs/2),(f3+30)/(fs/2), ], btype='bandstop', analog=False, output='ba')
    h = signal.lfilter(b3, a3, imp[:100])
    ax[1,0].stem(np.arange(sampleNumbers), h, basefmt= ' ')
    ax[1,0].set_title('Impulzna odozva pre f3')
    ax[1,0].set_xlabel('N')
        
    b4, a4 = signal.butter(4, [(f4-30)/(fs/2),(f4+30)/(fs/2)], btype='bandstop', analog=False, output='ba')
    h = signal.lfilter(b4, a4, imp[:100])
    ax[1,1].stem(np.arange(sampleNumbers), h, basefmt= ' ')
    ax[1,1].set_title('Impulzna odozva pre f4')
    ax[1,1].set_xlabel('N')


    plt.tight_layout()
    plt.show()



    ### Uloha 8.

    #z1, p1, k1 = signal.butter(4, [(f1-30)/(fs/2),(f1+30)/(fs/2)], btype='bandstop', analog=False, output='zpk')
    #z2, p2, k2 = signal.butter(4, [(f2-30)/(fs/2),(f2+30)/(fs/2)], btype='bandstop', analog=False, output='zpk')
    #z3, p3, k3 = signal.butter(4, [(f3-30)/(fs/2),(f3+30)/(fs/2)], btype='bandstop', analog=False, output='zpk')
    #z4, p4, k4 = signal.butter(4, [(f4-30)/(fs/2),(f4+30)/(fs/2)], btype='bandstop', analog=False, output='zpk')


    z1, p1, k1 = signal.tf2zpk(b1, a1)
    z2, p2, k2 = signal.tf2zpk(b2, a2)
    z3, p3, k3 = signal.tf2zpk(b3, a3)
    z4, p4, k4 = signal.tf2zpk(b4, a4)


    plt.figure(figsize=(6,6))
    # jednotkova kruznice
    ang = np.linspace(0, 2*np.pi,100)
    plt.plot(np.cos(ang), np.sin(ang))

    # nuly, poly
    plt.scatter(np.real(z1), np.imag(z1), marker='o', facecolors='none', edgecolors='r')
    plt.scatter(np.real(z2), np.imag(z2), marker='o', facecolors='none', edgecolors='r')
    plt.scatter(np.real(z3), np.imag(z3), marker='o', facecolors='none', edgecolors='r')
    plt.scatter(np.real(z4), np.imag(z4), marker='o', facecolors='none', edgecolors='r', label='nuly')
    plt.scatter(np.real(p1), np.imag(p1), marker='x', color='g')
    plt.scatter(np.real(p2), np.imag(p2), marker='x', color='g')
    plt.scatter(np.real(p3), np.imag(p3), marker='x', color='g')
    plt.scatter(np.real(p4), np.imag(p4), marker='x', color='g', label='póly')

    plt.gca().set_xlabel('Realná složka $\mathbb{R}\{$z$\}$')
    plt.gca().set_ylabel('Imaginarní složka $\mathbb{I}\{$z$\}$')

    plt.grid(alpha=0.5, linestyle='--')
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.show()

    




    ## Uloha 9
    fig, ax = plt.subplots(2, 1, figsize=(10,5))
    w1, H1 = signal.freqz(b1, a1)
    ax[0].plot(w1/2/np.pi*fs, np.abs(H1))
    ax[1].plot(w1/2 /np.pi*fs, np.angle(H1))
    

    w2, H2 = signal.freqz(b2, a2)
    ax[0].plot(w2/2/np.pi*fs, np.abs(H2))
    ax[1].plot(w2/2 /np.pi*fs, np.angle(H2))

    w3, H3 = signal.freqz(b3, a3)
    ax[0].plot(w3/2/np.pi*fs, np.abs(H3))
    ax[1].plot(w3/2 /np.pi*fs, np.angle(H3))

    w4, H4 = signal.freqz(b4, a4)
    ax[0].plot(w4/2/np.pi*fs, np.abs(H4))
    ax[1].plot(w4/2 /np.pi*fs, np.angle(H4))

    ax[0].set_xlabel('Frekvence [Hz]')
    ax[0].set_title('Modul frekvenční charakteristiky $|H(e^{j\omega})|$')

    ax[1].set_xlabel('Frekvence [Hz]')
    ax[1].set_title('Argument frekvenční charakteristiky $\mathrm{arg}\ H(e^{j\omega})$')
    plt.tight_layout()
    plt.show()

    
    ### Uloha 10

    dataFiltered = signal.filtfilt(b1 ,a1 , data)
    dataFiltered = signal.filtfilt(b2 ,a2 , dataFiltered)
    dataFiltered = signal.filtfilt(b3 ,a3 , dataFiltered)
    dataFiltered = signal.filtfilt(b4 ,a4 , dataFiltered)

    time = np.arange(dataFiltered.size)/fs
    plt.figure(figsize=(10, 5))
    plt.gca().set_xlabel('$t[s]$')
    plt.plot(time, data, label="Povodny signal")
    plt.plot(time, dataFiltered, color="red", label="Vyfiltrovany signal")
    plt.gca().set_title("Rozdiel medzi vyfiltrovanym a povodnym signalom")
    plt.legend(loc='upper left')
    plt.show()
    
    
    f, t, sgr = signal.spectrogram(dataFiltered, fs)
    sgr_log = 10 * np.log10(sgr+1e-20) 
    plt.figure(figsize=(10,5))
    plt.pcolormesh(t,f,sgr_log)
    plt.gca().set_xlabel('t [s]')
    plt.gca().set_title('Vyfiltrovany spektrogram')
    plt.gca().set_ylabel('f [Hz]')
    cbar = plt.colorbar()
    cbar.set_label('Spektralní hustota výkonu [dB]', rotation=270, labelpad=15)

    plt.tight_layout()
    plt.show()

    wavfile.write("audio/clean_bandstop.wav", fs, dataFiltered.astype(np.float32))

   





questOne()
nextQuests()

