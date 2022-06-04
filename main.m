clc
close all
clear all

% Maria B�k

filename = 'wav/25.wav';
outputfilename = 'wav/out/25.wav';

[data, samplerate] = audioread(filename);

t = length(data);
time = data / samplerate;


% 1. Model autoregresyjny rz�du r = 4
% y(t) = a1*y(t-1) + a2*y(t-2) + a3*y(t-3) +a4*y(t-4) + e
% 2. Algorytm Wa�onych Najmniejszych Kwadrat�w (EW-LS)
% -> identyfikacja parametr�w modelu AR(4)
lbda = 0.4;  % sta�a zapominania
M = 20;
od = 0;

new_data = data;
noise_detected = zeros(t, 1);

for i = 6+M:t

    R = zeros(4);
    p = zeros(4, 1);
   
     for j = 0:M
         w = lbda ^ j; % okno wyk�adnicze
         R = R + w * fi(data, i-j) * transpose(fi(data, i-j)); % macierz regresji
         p = p + w * data(i-j) * fi(data, i-j);
     end
     
     if det(R) ~= 0 % macierz regresji musi by� odwracalna ==> warunek identyfikowalno�ci modelu AR
        theta = inv(R) * p; % og�lna posta� wa�onego estymatora najmniejszych kwadrat�w #a1 a3 a3 a4
        % 3. Detektor zak��ce� impulsowych kwestionuj�cy w ka�dym kroku algorytmu EW-LS pr�bki sygna�u, dla kt�rych
        %    bezwzgl�dna warto�� b��du predykcji prekracza trzykrotnie lokaln� warto�� �redniego odchylenia
        %    standardowego b��du resztowego
        e(i) = data(i) - transpose(fi(data, i)) * theta; % b��d predykcji
        
        n = 0;
        for j = 1:M
            n = n + (data(i-j) - transpose(fi(data, i-j)) * theta)^2; % b��d resztowy
        end
        od = sqrt(1/M * n); % odchylenie standardowe b��du resztowego
        
        if n > 3 * od
            noise_detected(i) = 1;
        else
            noise_detected(i) = 0;
        end
            
        good_prev = 0;
        good_after = 0;
        
        % sprawd� 4 pr�bki wstecz
        if noise_detected(i) == 1
            for k = 1:4
                if noise_detected(i-k) == 0
                    good_prev = new_data(i-k);
                    break
                end
            end
        end
        
        % sprawd� 4 pr�bki wprz�d
        for k=1:4
        	if i <= t-4
                if noise_detected(i) == 1
                    if noise_detected(i+k) == 0
                        good_prev = new_data(i+k);
                        break
                    end
                end
            else
                break
            end
        end
        
        % 4. Metoda interpolacji liniowej
        if noise_detected(i) == 1
            new_data(i) = 0.5 * (good_prev + good_after); % podmiana pr�bki
        end
        
     end
end


audiowrite(outputfilename, new_data, samplerate);

function f = fi(d, x)
    f =  [d(x-1); d(x-2); d(x-3); d(x-4)];
end