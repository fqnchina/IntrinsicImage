require 'torch'
require 'image'
require 'sys'
require 'cunn'
require 'cutorch'
require 'cudnn'
require 'nngraph'

imgPath = '/raid/qingnan/data/MPI-main-clean/'
saveDir = '/raid/qingnan/codes/intrinsic/MPI_main_SceneSplit_front_combine/'

local file = '/raid/qingnan/data/MPI_main_sceneSplit-fullsize-NoDefect-test.txt'
local files = {}
local f = io.open(file, "rb")
while true do
  local line = f:read()
  if line == nil then break end
  table.insert(files, line)
end
f:close()
local testsetSize = #files

model = torch.load('/raid/qingnan/codes/intrinsic/netfiles/model_MPI_main_SceneSplit_front_combine_best.net')
model = model:cuda()
model:training()

for _,file in ipairs(files) do

	local tempInput = image.load(file)
	local height = tempInput:size(2)
	local width = tempInput:size(3)
	local savColor = string.gsub(file,imgPath,saveDir)

	local labelAFile = string.gsub(file,'clean','albedo')
	local labelSFile = string.gsub(file,'clean','shading')
	local templabelA = image.load(labelAFile)
	local templabelS = image.load(labelSFile)

	local savLabelA = string.gsub(savColor,'.png','-label-albedo.png')
	local savLabelS = string.gsub(savColor,'.png','-label-shading.png')
	local savAlbedo = string.gsub(savColor,'.png','-predict-albedo.png')
	local savShading = string.gsub(savColor,'.png','-predict-shading.png')
	local savColor = string.gsub(savColor,'.png','-input.png')

	local input = torch.CudaTensor(1, 3, height, width)
	local labelA = torch.CudaTensor(1, 3, height, width)
	local labelS = torch.CudaTensor(1, 3, height, width)
	input[1] = tempInput
	labelA[1] = templabelA
	labelS[1] = templabelS

	image.save(savColor,input[1])
	image.save(savLabelA,labelA[1])
	image.save(savLabelS,labelS[1])

	input = input * 255

	local predictions = model:forward(input)
	predictionsA = predictions[8]
	predictionsS = predictions[1]

	for m = 1,3 do
	 local numerator = torch.dot(predictionsA[1][m], labelA[1][m])
	 local denominator = torch.dot(predictionsA[1][m], predictionsA[1][m])
	 local alpha = numerator/denominator
	 predictionsA[1][m] = predictionsA[1][m] * alpha
	end

	for m = 1,3 do
	 local numerator = torch.dot(predictionsS[1][m], labelS[1][m])
	 local denominator = torch.dot(predictionsS[1][m], predictionsS[1][m])
	 local alpha = numerator/denominator
	 predictionsS[1][m] = predictionsS[1][m] * alpha
	end

	image.save(savAlbedo,predictionsA[1])
	image.save(savShading,predictionsS[1])
end