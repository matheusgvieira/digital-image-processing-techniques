clear, close all, clc

img2 = rgb2gray(imread('../images/highlight/inputs/statue.jpeg'));
figure, imshow(img2)


Img2 = fftshift(fft2(img2));
Img2_m = abs(Img2);
Img2_f = angle(Img2);

##figure, imshow(log(Img2_m),[]);
##figure, imshow(Img2_f,[]);

img2i = abs(ifft2(ifftshift(Img2)));

%% Filtragem sem padding
[M,N] = size(img2);
[u,v] = meshgrid(1:N,1:M);
D = sqrt((u-M/2).^2 + (v-N/2).^2);
 
gamaL = 0.5;
gamaH = 2;
c = 1;
D0 = 80;

Hlp = (gamaH - gamaL)*(1 - exp(-c*((D.^2)/(D0.^2)))) + gamaL;
##Hlp = exp(-D.^2/(2*D0^2));

##figure, imshow(Hlp,[])
##figure, imshow(log(Hlp),[])
##figure, mesh(H)

Im2f_m = Img2_m.*Hlp;

##figure, imshow(log(Im2f_m),[]);

j = sqrt(-1);
Im2f = Im2f_m.*exp(j*Img2_f);

img2fi = uint8(real(ifft2(ifftshift(Im2f))));

figure, imshow(img2fi)