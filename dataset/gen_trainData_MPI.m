clear all;
patch_dim = 300;

root_source = '/raid/qingnan/data/MPI-main-shading/';

root_input = '/raid/qingnan/data/MPI-main-input-300';
root_albedo = '/raid/qingnan/data/MPI-main-albedo-300';
root_shading = '/raid/qingnan/data/MPI-main-shading-300';
root_mask = '/raid/qingnan/data/MPI-main-mask-300';

if exist(root_input, 'dir')
    rmdir(root_input,'s');
end
mkdir(root_input);

if exist(root_albedo, 'dir')
    rmdir(root_albedo,'s');
end
mkdir(root_albedo);

if exist(root_shading, 'dir')
    rmdir(root_shading,'s');
end
mkdir(root_shading);

if exist(root_mask, 'dir')
    rmdir(root_mask,'s');
end
mkdir(root_mask);

firstList = dir([root_source '*.png']);
parfor m = 1:length(firstList)
    shadingName = fullfile(root_source,firstList(m).name);                      
    inputName = strrep(shadingName,'shading','clean');
    albedoName = strrep(shadingName,'shading','albedo');
    maskName = strrep(shadingName,'shading','mask');

    inputImg = imread(inputName);
    albedoImg = imread(albedoName);
    shadingImg = imread(shadingName);
    maskImg = imread(maskName);

    [height,width,channel] = size(inputImg);
    if height < patch_dim || width < patch_dim
        continue;
    end
    if channel == 1
        continue;
    end

    count = 0;
    while count < 10
        outInput = sprintf('%s/%s-%d.png',root_input,firstList(m).name(1:end-4),count+1);
        outAlbedo = sprintf('%s/%s-%d.png',root_albedo,firstList(m).name(1:end-4),count+1);
        outShading = sprintf('%s/%s-%d.png',root_shading,firstList(m).name(1:end-4),count+1);
        outMask = sprintf('%s/%s-%d.png',root_mask,firstList(m).name(1:end-4),count+1);

        y = randi([1 height - patch_dim + 1],1);
        x = randi([1 width - patch_dim + 1],1);

        ranNum = randi([1,2],1);

        patchMask = im2double(maskImg(y:y+patch_dim-1,x:x+patch_dim-1,:));
        maskNum = eq(patchMask,0);
        if sum(maskNum(:)) > 0.05 * patch_dim * patch_dim
            continue;
        end

        if ranNum == 1
            patchMask = flipdim(patchMask ,2);
        end
        imwrite(patchMask,outMask);

        patchInput = im2double(inputImg(y:y+patch_dim-1,x:x+patch_dim-1,:));
        if ranNum == 1
            patchInput = flipdim(patchInput ,2);
        end
        imwrite(patchInput,outInput);

        patchAlbedo = im2double(albedoImg(y:y+patch_dim-1,x:x+patch_dim-1,:));
        if ranNum == 1
            patchAlbedo = flipdim(patchAlbedo ,2);
        end
        imwrite(patchAlbedo,outAlbedo);

        patchShading = im2double(shadingImg(y:y+patch_dim-1,x:x+patch_dim-1,:));
        if ranNum == 1
            patchShading = flipdim(patchShading ,2);
        end
        imwrite(patchShading,outShading);

        count = count + 1;
    end
end