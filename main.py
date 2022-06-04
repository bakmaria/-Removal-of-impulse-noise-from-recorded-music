#Maria Bąk
from scipy.io.wavfile import read, write
import numpy as np
from math import sqrt


def fi(x):
    temp = [[data[x-1]], [data[x-2]], [data[x-3]], [data[x-4]]]
    temp = np.array(temp)
    return temp


# Nagranie przed obróbką
filename = 'wav/01d.wav'

#wave_obj = sa.WaveObject.from_wave_file(filename)
#play_obj = wave_obj.play()
#play_obj.wait_done()

samplerate, data = read(filename)
duration = len(data)/samplerate

time = np.arange(0, duration, 1/samplerate)

# 1. Model autoregresyjny rzędu r = 4
# y(t) = a1*y(t-1) + a2*y(t-2) + a3*y(t-3) +a4*y(t-4) + e
# 2. Algorytm Ważonych Najmniejszych Kwadratów (EW-LS)
# -> identyfikacja parametrów modelu AR(4)
lbda = 0.4  # stała zapominania
t = len(data)
theta_prev = 0
M = 20
count = 0
od = 0

new_data = data.copy()
noise_data = [0] * len(data)

noise = []
noise_numbers = []
noise_count = 0

noise_detected = [0] * len(data)

for i in range(6+M, t):
    print(i)

    R = 0
    p = 0

    for j in range(M):
        w = pow(lbda, j)  # okno wykładnicze
        R += w * np.matmul(fi(i-j), fi(i-j).transpose())  # macierz regresji #macierz 4x4
        p += w * data[i-j] * fi(i-j)  # macierz 4x1

    if np.linalg.det(R) != 0:   # macierz regresji musi być odwracalna ==> warunek identyfikowalności modelu AR
        theta = np.matmul(np.linalg.inv(R), p)  # ogólna postać ważonego estymatora najmniejszych kwadratów #a1 a3 a3 a4
        # 3. Detektor zakłóceń impulsowych kwestionujący w każdym kroku algorytmu EW-LS próbki sygnału, dla których
        #    bezwzględna wartość błędu predykcji prekracza trzykrotnie lokalną wartość średniego odchylenia
        #    standardowego błędu resztowego
        e = data[i] - np.matmul(fi(i).transpose(), theta)   # błąd predykcji

        n = 0
        for j in range(1, M):
            n += pow(data[i - j] - np.matmul(fi(i - j).transpose(), theta), 2)   # błąd resztowy
        od = sqrt(1/M * n)  # odchylenie standardowe błędu resztowego

        if n > 3 * od:
            noise_detected[i] = 1
        else:
            noise_detected[i] = 0

        good_prev = 0
        good_after = 0

        # sprawdz 4 próbki wstecz
        if noise_detected[i] == 1:
            for k in range(1, 4):
                if noise_detected[i - k] == 0:
                    good_prev = data[i - k]
                    break

        # sprawdz 4 próbki wprzod
        if i <= t - 4:
            if noise_detected[i] == 1:
                for k in range(1, 4):
                    if noise_detected[i + k] == 0:
                        good_prev = data[i + k]
                        break
        else:
            break


        # 4. Metoda interpolacji liniowej
        if noise_detected[i] == 1:
            new_data[i] = 0.5 * (good_prev + good_after)  # podmiana próbki

'''
plt.subplot(2, 2, 1)
plt.plot(time, data, 'b')
plt.ylim([-30000, 30000])
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.title('Przed usunięciem zakłóceń\n%s' % filename)

plt.subplot(2, 2, 2)
plt.plot(time, noise_data, 'g')
plt.ylim([-30000, 30000])
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.title('Szum')

plt.subplot(2, 2, 3)
plt.plot(time, new_data, 'r')
plt.ylim([-30000, 30000])
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.title('Po usunięciu zakłóceń\n%s' % filename)

plt.subplot(2, 2, 4)
plt.plot(time, data, 'b')
plt.plot(time, new_data, 'r')
plt.ylim([-30000, 30000])
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.title('Nałożone sygnały')

plt.tight_layout()
plt.show()
'''
outputfile = filename.split('.')[0] + '-out.wav'
write(outputfile, samplerate, new_data)
#wave_obj2 = sa.WaveObject.from_wave_file(outputfile)
#play_obj2 = wave_obj2.play()
#play_obj2.wait_done()
print(outputfile)







