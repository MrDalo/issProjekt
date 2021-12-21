import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import soundfile as sf


#nacitanie signalu
data, fs = sf.read('xkrali20.wav')


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

fs, data = wavfile.read('xkrali20.wav')

#ustrednenie
data = data - np.mean(data)


#normalizacia input signalu
data = data / 2**15
matrixOfData = np.zeros(shape=(round(data.size/512)-2, 1024))


for i in range(round(data.size/512)-2):
    arrayOfData = data[i*512:((i+1)*512)+512]

    matrixOfData[i]=arrayOfData
    #plt.figure(figsize=(10, 5))
    time = np.arange(arrayOfData.size)/fs
    #plt.plot(time, arrayOfData)
    #plt.gca().set_title(f"Predspracovanie signalu, frame: {i}")
    #plt.show()


    
print(matrixOfData)




#plt.figure(figsize=(10, 5))
#time = np.arange(data.size)/fs
#plt.plot(time, data)
#plt.show()



