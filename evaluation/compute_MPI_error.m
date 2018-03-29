totalMSEA = 0;
totalLMSEA = 0;
totalDSSIMA = 0;
totalMSES = 0;
totalLMSES = 0;
totalDSSIMS = 0;
count = 0;

inputDir = '/raid/qingnan/codes/intrinsic/MPI_main_SceneSplit_front_combine/';
s = dir([inputDir '*-input.png']);
for n = 1:length(s)
    albedoName = [inputDir s(n,1).name(1:end-10) '-predict-albedo.png'];
    shadingName = [inputDir s(n,1).name(1:end-10) '-predict-shading.png'];
    labelAlbedoName = [inputDir s(n,1).name(1:end-10) '-label-albedo.png'];
    labelShadingName = [inputDir s(n,1).name(1:end-10) '-label-shading.png'];

    albedo = im2double(imread(albedoName));
    labelAlbedo = im2double(imread(labelAlbedoName));
    shading = im2double(imread(shadingName));
    labelShading = im2double(imread(labelShadingName));
    [height, width, channel] = size(albedo);

    totalMSEA = totalMSEA + evaluate_one_k(albedo,labelAlbedo);
    totalLMSEA = totalLMSEA + levaluate_one_k(albedo,labelAlbedo);
    totalDSSIMA = totalDSSIMA + (1-evaluate_ssim_one_k_fast(albedo,labelAlbedo))/2;

    totalMSES = totalMSES + evaluate_one_k(shading,labelShading);
    totalLMSES = totalLMSES + levaluate_one_k(shading,labelShading);
    totalDSSIMS = totalDSSIMS + (1-evaluate_ssim_one_k_fast(shading,labelShading))/2;

    count = count + 1;
    disp(count);
end
totalMSEA = totalMSEA/count;
totalLMSEA = totalLMSEA/count;
totalDSSIMA = totalDSSIMA/count;
totalMSES = totalMSES/count;
totalLMSES = totalLMSES/count;
totalDSSIMS = totalDSSIMS/count;
disp(sprintf('albedo: mse: %f, lmse: %f, dssim: %f',totalMSEA,totalLMSEA,totalDSSIMA));
disp(sprintf('shading: mse: %f, lmse: %f, dssim: %f',totalMSES,totalLMSES,totalDSSIMS));