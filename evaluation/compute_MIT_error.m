clear all;
inputDir = '/raid/qingnan/codes/intrinsic/MIT_combine/';
images = dir([inputDir '*-input.png']);
mse_albedo = {};
mse_shading = {};
lmse = {};

for m =1:length(images)
    filename = [inputDir images(m).name];
    albedoname_predict = [inputDir images(m).name(1:end-10) '-predict-albedo.png'];
    shadingname_predict = [inputDir images(m).name(1:end-10) '-predict-shading.png'];
    albedoname_label = [inputDir images(m).name(1:end-10) '-label-albedo.png'];
    shadingname_label = [inputDir images(m).name(1:end-10) '-label-shading.png'];
    maskname_label = [inputDir images(m).name(1:end-10) '-label-mask.png'];
    
    albedo_predict = im2double(imread(albedoname_predict));
    shading_predict = im2double(imread(shadingname_predict));
    albedo_label = im2double(imread(albedoname_label));
    shading_label = im2double(imread(shadingname_label));
    mask = (imread(maskname_label));
    V = mask > 0;

    V3 = repmat(V,[1,1,size(shading_label,3)]);  
    
    errs_grosse = nan(1, size(albedo_label,3));
    for c = 1:size(albedo_label,3)
      errs_grosse(c) = 0.5 * MIT_mse(shading_predict(:,:,c), shading_label(:,:,c), V) + 0.5 * MIT_mse(albedo_predict(:,:,c), albedo_label(:,:,c), V);
    end
    lmse{m} = mean(errs_grosse);
    
    alpha_shading = sum(shading_label(V3) .* shading_predict(V3)) ./ max(eps, sum(shading_predict(V3) .* shading_predict(V3)));
    S = shading_predict * alpha_shading;

    alpha_reflectance = sum(albedo_label(V3) .* albedo_predict(V3)) ./ max(eps, sum(albedo_predict(V3) .* albedo_predict(V3)));
    A = albedo_predict * alpha_reflectance;

    mse_shading{m} =  mean((S(V3) - shading_label(V3)).^2);
    mse_albedo{m} =  mean((A(V3) - albedo_label(V3)).^2);
end

ave_lmse = 0;
ave_mse_albedo = 0;
ave_mse_shading = 0;
for m =1:length(images)
    ave_lmse = ave_lmse + log(lmse{m});
    ave_mse_albedo = ave_mse_albedo + log(mse_albedo{m});
    ave_mse_shading = ave_mse_shading + log(mse_shading{m});
end
ave_lmse = exp(ave_lmse/length(images))
ave_mse_albedo = exp(ave_mse_albedo/length(images))
ave_mse_shading = exp(ave_mse_shading/length(images))