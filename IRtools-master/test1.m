%% test: 图片模糊和加噪声

IRtools_setup;
rng(0);  % 伪随机数种子
clear, clc;

%% create data
% image 1
optblur1 = PRset('trueImage', 'satellite','BlurLevel','medium','BC', 'zero','CommitCrime', 'on');
n1 = 128;
[A1, b1_true, x1_true, ProbInfo1] = PRblurspeckle(n1,optblur1);

% image 2
load Grain
n2 = size(x_true,1);
optblur2.trueImage = reshape(x_true,n2,n2);
optblur2.BlurLevel = 'mild';
optblur2.CommitCrime = 'on';
optblur2.BC = 'reflective';
[A2, b2_true, x_true, ProbInfo2] = PRblurgauss(n2,optblur2);


% add noise
nel = 0.01; % Noise level
b1 = PRnoise(b1_true, 'gauss', nel);  % Observed Image
b2 = PRnoise(b2_true, 'gauss', nel);  % Observed Image

% PRshowx(ProbInfo2.psf,ProbInfo2);
%mesh(ProbInfo1.psf);

%%
figure;
subplot(2, 2, 1), imshow(reshape(x1_true(:), n1, n1), []), title('true image');
subplot(2, 2, 2), imshow(reshape(b1, n1, n1), []), title('blurred and noisy image');
subplot(2, 2, 3), imshow(reshape(x_true(:), n2, n2), []), title('true image');
subplot(2, 2, 4), imshow(reshape(b2, n2, n2), []), title('blurred and noisy image');
