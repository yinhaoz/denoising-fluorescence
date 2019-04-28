clc
clear
close all

data_name = 'Confocal_BPAE_B';

addpath('./code')
addpath('../utils')
dir_avg1 = ['../../dataset/', data_name, '/raw/19/']; % 50 images
dir_avg2 = ['../../dataset/', data_name, '/avg2/19/']; % 50 images
dir_avg4 = ['../../dataset/', data_name, '/avg4/19/']; % 50 images
dir_avg8 = ['../../dataset/', data_name, '/avg8/19/']; % 50 images
dir_avg16 = ['../../dataset/', data_name, '/avg16/19/']; % 50 images
dir_gt = ['../../dataset/', data_name, '/gt/19/']; % 1 image


[avg1_array] = import_img_array(dir_avg1);
[avg2_array] = import_img_array(dir_avg2);
[avg4_array] = import_img_array(dir_avg4);
[avg8_array] = import_img_array(dir_avg8);
[avg16_array] = import_img_array(dir_avg16);

[img_gt] = import_img_array(dir_gt);


%%%%%%%%%%%%%%%%%%%%%%%%%%% noise realizations %%%%%%%%%%%%%%%%%%%%%%%%%%%
n_repeat = 50;
% n_repeat = 1;

diary([data_name, '.log'])

PSNR_avg1 = zeros(n_repeat, 1);
PSNR_avg2 = zeros(n_repeat, 1);
PSNR_avg4 = zeros(n_repeat, 1);
PSNR_avg8 = zeros(n_repeat, 1);
PSNR_avg16 = zeros(n_repeat, 1);

SSIM_avg1 = zeros(n_repeat, 1);
SSIM_avg2 = zeros(n_repeat, 1);
SSIM_avg4 = zeros(n_repeat, 1);
SSIM_avg8 = zeros(n_repeat, 1);
SSIM_avg16 = zeros(n_repeat, 1);

TIME_avg = zeros(n_repeat, 1);



for i_repeat = 1:n_repeat
    
    fprintf('----- The %d / %d realization -----\n', i_repeat,n_repeat)
    
    % extract the images and average them
    img_avg1 = avg1_array(:,:,i_repeat);
    img_avg2 = avg2_array(:,:,i_repeat);
    img_avg4 = avg4_array(:,:,i_repeat);
    img_avg8 = avg8_array(:,:,i_repeat);
    img_avg16 = avg16_array(:,:,i_repeat);   
    
   
    % perform the denoising algorithm
    
    [denoise_avg1, time_avg1] = denoise_VST_NLM(img_avg1);
    [denoise_avg2, time_avg2] = denoise_VST_NLM(img_avg2);
    [denoise_avg4, time_avg4] = denoise_VST_NLM(img_avg4);
    [denoise_avg8, time_avg8] = denoise_VST_NLM(img_avg8);
    [denoise_avg16, time_avg16] = denoise_VST_NLM(img_avg16);

    TIME_avg(i_repeat) = mean([time_avg1, time_avg2, ...
        time_avg4, time_avg8, time_avg16]);
    
    % compute the PSNR
    PSNR_avg1(i_repeat) = 10*log10(1/mean((img_gt(:)-denoise_avg1(:)).^2));
    PSNR_avg2(i_repeat) = 10*log10(1/mean((img_gt(:)-denoise_avg2(:)).^2));
    PSNR_avg4(i_repeat) = 10*log10(1/mean((img_gt(:)-denoise_avg4(:)).^2));
    PSNR_avg8(i_repeat) = 10*log10(1/mean((img_gt(:)-denoise_avg8(:)).^2));
    PSNR_avg16(i_repeat) = 10*log10(1/mean((img_gt(:)-denoise_avg16(:)).^2));    
    
    % compute the SSIM
    SSIM_avg1(i_repeat) = ssim(denoise_avg1, img_gt);
    SSIM_avg2(i_repeat) = ssim(denoise_avg2, img_gt);
    SSIM_avg4(i_repeat) = ssim(denoise_avg4, img_gt);
    SSIM_avg8(i_repeat) = ssim(denoise_avg8, img_gt);
    SSIM_avg16(i_repeat) = ssim(denoise_avg16, img_gt);    

end

fprintf('Average = 1 (PSNR: %.2f dB, SSIM: %.4f)\n', mean(PSNR_avg1), mean(SSIM_avg1))
fprintf('Average = 2 (PSNR: %.2f dB, SSIM: %.4f)\n', mean(PSNR_avg2), mean(SSIM_avg2))
fprintf('Average = 4 (PSNR: %.2f dB, SSIM: %.4f)\n', mean(PSNR_avg4), mean(SSIM_avg4))
fprintf('Average = 8 (PSNR: %.2f dB, SSIM: %.4f)\n', mean(PSNR_avg8), mean(SSIM_avg8))
fprintf('Average = 16 (PSNR: %.2f dB, SSIM: %.4f)\n', mean(PSNR_avg16), mean(SSIM_avg16))
fprintf('Average time for each denoising operation = %.2f\n', mean(TIME_avg))


figure;
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0, 0.8, 1]);

subplot(3, 4, 1)
imagesc(img_avg1); axis tight equal; set(gca, 'XTick', [], 'YTick', []);
title('Raw x1')
subplot(3, 4, 2)
imagesc(denoise_avg1); axis tight equal; set(gca, 'XTick', [], 'YTick', []);
title('Denoise x1')

subplot(3, 4, 3)
imagesc(img_avg2); axis tight equal; set(gca, 'XTick', [], 'YTick', []);
title('Raw x2')
subplot(3, 4, 4)
imagesc(denoise_avg2); axis tight equal; set(gca, 'XTick', [], 'YTick', []);
title('Denoise x2')

subplot(3, 4, 5)
imagesc(img_avg4); axis tight equal; set(gca, 'XTick', [], 'YTick', []);
title('Raw x4')
subplot(3, 4, 6)
imagesc(denoise_avg4); axis tight equal; set(gca, 'XTick', [], 'YTick', []);
title('Denoise x4')

subplot(3, 4, 7)
imagesc(img_avg8); axis tight equal; set(gca, 'XTick', [], 'YTick', []);
title('Raw x8')
subplot(3, 4, 8)
imagesc(denoise_avg8); axis tight equal; set(gca, 'XTick', [], 'YTick', []);
title('Denoise x8')

subplot(3, 4, 9)
imagesc(img_avg16); axis tight equal; set(gca, 'XTick', [], 'YTick', []);
title('Raw x16')
subplot(3, 4, 10)
imagesc(denoise_avg16); axis tight equal; set(gca, 'XTick', [], 'YTick', []);
title('Denoise x16')

subplot(3, 4, 12)
imagesc(img_gt); axis tight equal; set(gca, 'XTick', [], 'YTick', []);
title('Ground Truth (x50)')

colormap(gray);
saveas(gcf,[data_name,'.png'])

diary off






