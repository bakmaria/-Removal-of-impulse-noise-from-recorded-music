clc
close all
clear all

% Maria B¹k

filename = 'wav/25.wav';
outputfilename = 'wav/out/25.wav';

[data, samplerate] = audioread(filename);

t = length(data);
time = data / samplerate;


% 1. Model autoregresyjny rzêdu r = 4
% y(t) = a1*y(t-1) + a2*y(t-2) + a3*y(t-3) +a4*y(t-4) + e
% 2. Algorytm Wa¿onych Najmniejszych Kwadratów (EW-LS)
% -> identyfikacja parametrów modelu AR(4)
lbda = 0.4;  % sta³a zapominania
M = 20;
od = 0;

new_data = data;
noise_detected = zeros(t, 1);

for i = 6+M:t

    R = zeros(4);
    p = zeros(4, 1);
   
     for j = 0:M
         w = lbda ^ j; % okno wyk³adnicze
         R = R + w * fi(data, i-j) * transpose(fi(data, i-j)); % macierz regresji
         p = p + w * data(i-j) * fi(data, i-j);
     end
     
     if det(R) ~= 0 % macierz regresji musi byæ odwracalna ==> warunek identyfikowalnoœci modelu AR
        theta = inv(R) * p; % ogólna postaæ wa¿onego estymatora najmniejszych kwadratów #a1 a3 a3 a4
        % 3. Detektor zak³óceñ impulsowych kwestionuj¹cy w ka¿dym kroku algorytmu EW-LS próbki sygna³u, dla których
        %    bezwzglêdna wartoœæ b³êdu predykcji prekracza trzykrotnie lokaln¹ wartoœæ œredniego odchylenia
        %    standardowego b³êdu resztowego
        e(i) = data(i) - transpose(fi(data, i)) * theta; % b³¹d predykcji
        
        n = 0;
        for j = 1:M
            n = n + (data(i-j) - transpose(fi(data, i-j)) * theta)^2; % b³¹d resztowy
        end
        od = sqrt(1/M * n); % odchylenie standardowe b³êdu resztowego
        
        if n > 3 * od
            noise_detected(i) = 1;
        else
            noise_detected(i) = 0;
        end
            
        good_prev = 0;
        good_after = 0;
        
        % sprawdŸ 4 próbki wstecz
        if noise_detected(i) == 1
            for k = 1:4
                if noise_detected(i-k) == 0
                    good_prev = new_data(i-k);
                    break
                end
            end
        end
        
        % sprawdŸ 4 próbki wprzód
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
            new_data(i) = 0.5 * (good_prev + good_after); % podmiana próbki
        end
        
     end
end


audiowrite(outputfilename, new_data, samplerate);

function f = fi(d, x)
    f =  [d(x-1); d(x-2); d(x-3); d(x-4)];
end