require 'torch'
require 'image'
require 'sys'
require 'cunn'
require 'cutorch'
require 'cudnn'
require 'nngraph'
cudnn.fastest = true
cudnn.benchmark = true

imgPath = '/raid/qingnan/data/iiw%-dataset/data_even'
savePath = '/raid/qingnan/codes/intrinsic/IIW_combine'

-- NOTE: the test images have to be cropped to the size of even number to fit our network structure. 

local file = '/raid/qingnan/data/iiw_Learning_Lightness_test.txt'
local files = {}
local f = io.open(file, "rb")
while true do
  local line = f:read()
  if line == nil then break end
  table.insert(files, line)
end
f:close()
local testsetSize = #files

model_edge_input = nn.EdgeComputation(100)
model = torch.load('/raid/qingnan/codes/intrinsic/netfiles/model_IIW_combine_best.net')
model = model:cuda()
model:training()

for _,inputFile in ipairs(files) do

	local inputImg = image.load(inputFile)
	local savColor = string.gsub(inputFile,imgPath,savePath)
	image.save(savColor,inputImg)
	local height = inputImg:size(2)
	local width = inputImg:size(3)

	local input = torch.CudaTensor(1, 3, height, width)
	input[1] = inputImg:cuda()
	input = input * 255

	-- local guide_albedo_file = string.gsub(inputFile,imgPath,'/raid/qingnan/data/iiw_L1_IntrinsicDecomposition_r/')
	-- local guide_a = image.load(guide_albedo_file)
	-- local guide_albedo = torch.CudaTensor(1, 3, height, width)
	-- guide_albedo[1] = guide_a:cuda()
	-- guide_albedo_mean = torch.sum(guide_albedo,2)/3

	-- local guide_shading_file = string.gsub(inputFile,imgPath,'/raid/qingnan/data/iiw_L1_IntrinsicDecomposition_s/')
	-- local guide_s = image.load(guide_shading_file)
	-- local guide_shading = torch.CudaTensor(1, 3, height, width)
	-- guide_shading[1] = guide_s:cuda()
	-- guide_shading = torch.sum(guide_shading,2)/3

	local guide_albedo = input/255
	local guide_shading = input/255
	local guide_albedo_mean = torch.sum(guide_albedo,2)/3

	local predictions_final = model:forward(input)
	predictions_final1 = predictions_final[1]
	predictions_final2 = predictions_final[2]
	predictions_final3 = predictions_final[3]

	-- uncomment the following line while testing the jointly trained model
	-- predictions_final1 = torch.cmax(torch.sum(predictions_final1,2)/3,0.0000000001)

	local r_value = torch.cmax(predictions_final1,0.0000000001)
	local input_mean = torch.cmax(torch.sum(input,2)/3,0.0000000001)
	local r_div = torch.cdiv(r_value,input_mean)

	local output_reflectance = torch.CudaTensor(1, 3, height, width)
	output_reflectance[{{},{1},{},{}}] = torch.cmul(input[{{},{1},{},{}}],r_div)
	output_reflectance[{{},{2},{},{}}] = torch.cmul(input[{{},{2},{},{}}],r_div)
	output_reflectance[{{},{3},{},{}}] = torch.cmul(input[{{},{3},{},{}}],r_div)

	for m = 1,3 do
	  local numerator = torch.dot(output_reflectance[1][m], guide_albedo[1][m])
	  local denominator = torch.dot(output_reflectance[1][m], output_reflectance[1][m])
	  local alpha = numerator/denominator
	  output_reflectance[1][m] = output_reflectance[1][m] * alpha
	end

	local sav = string.gsub(savColor,'.png','-r_prime.png')
	image.save(sav,output_reflectance[1])

	for m = 1,1 do
	  local numerator = torch.dot(predictions_final1[1][m], guide_albedo_mean[1][m])
	  local denominator = torch.dot(predictions_final1[1][m], predictions_final1[1][m])
	  local alpha = numerator/denominator
	  predictions_final1[1][m] = predictions_final1[1][m] * alpha
	end

	local sav = string.gsub(savColor,'.png','-r_small.png')
	image.save(sav,predictions_final1[1])

	for m = 1,3 do
	  local numerator = torch.dot(predictions_final3[1][m], guide_albedo[1][m])
	  local denominator = torch.dot(predictions_final3[1][m], predictions_final3[1][m])
	  local alpha = numerator/denominator
	  predictions_final3[1][m] = predictions_final3[1][m] * alpha
	end
	local predictions_final3 = torch.cmax(predictions_final3,0.0000000001)

	local sav = string.gsub(savColor,'.png','-R.png')
	image.save(sav,predictions_final3[1])

	input_mean = torch.sum(input,2)/3
	predictions_final3_mean = torch.sum(predictions_final3,2)/3
	local output_shading = torch.CudaTensor(1, 1, height, width)
	output_shading = torch.cdiv(input_mean,predictions_final3_mean)

	for m = 1,1 do
	  local numerator = torch.dot(output_shading[1][m], guide_shading[1][m])
	  local denominator = torch.dot(output_shading[1][m], output_shading[1][m])
	  local alpha = numerator/denominator
	  output_shading[1][m] = output_shading[1][m] * alpha
	end

	local sav = string.gsub(savColor,'.png','-S.png')
	image.save(sav,output_shading[1])


	input_edge = model_edge_input:forward(input)
	input_edge = input_edge/input_edge:max()
	input_edge = 1 - input_edge
	input_edge = input_edge/input_edge:max()
	local sav = string.gsub(savColor,'.png','-guidance_input.png')
	image.save(sav,input_edge[1])

	predictions_final2 = predictions_final2/predictions_final2:max()
	predictions_final2 = 1 - predictions_final2
	predictions_final2 = predictions_final2/predictions_final2:max()
	local sav = string.gsub(savColor,'.png','-guidance.png')
	image.save(sav,predictions_final2[1])
end
