require 'nn'
require 'optim'
require 'torch'
require 'cutorch'
require 'cunn'
require 'image'
require 'sys'
require 'nngraph'
require 'cudnn'
cudnn.fastest = true
cudnn.benchmark = true

--GPU 3
model_guidance = torch.load('/raid/qingnan/codes/intrinsic/netfiles/model_MIT_guidance_best_joint.net')
model_intrinsic = torch.load('/raid/qingnan/codes/intrinsic/netfiles/model_MIT_direct_best_joint.net')

h0 = nn.Identity()()
r = h0 - model_intrinsic
g = h0 - model_guidance
S = r - nn.SelectTable(1)
S_x = r - nn.SelectTable(2)
S_y = r - nn.SelectTable(3)
R = r - nn.SelectTable(4)
R_x = r - nn.SelectTable(5)
R_y = r - nn.SelectTable(6)
R_final = {R,g} - nn.JoinTable(2) - nn.DomainTransform(5,0.25,30,0)

model = nn.gModule({h0},{S,S_x,S_y,R,R_x,R_y,g,R_final})
model = model:cuda()

criterion = nn.ParallelCriterion():add(nn.MSECriterion(),0.2):add(nn.MSECriterion(),0.35):add(nn.MSECriterion(),0.35):add(nn.MSECriterion(),0.2):add(nn.MSECriterion(),0.35):add(nn.MSECriterion(),0.35):add(nn.MSECriterion(),0.35):add(nn.MSECriterion(),0.2)
criterion = criterion:cuda()
criterion_test = nn.ParallelCriterion():add(nn.MSECriterion(),0.5):add(nn.MSECriterion(),0.5)
criterion_test = criterion_test:cuda()

model_edge = nn.EdgeComputation(100)


postfix = 'MIT_combine'
max_iters = 50
batch_size = 1

model:training()
collectgarbage()

parameters, gradParameters = model:getParameters()

sgd_params = {
  learningRate = 1e-2,
  learningRateDecay = 1e-8,
  weightDecay = 0.0005,
  momentum = 0.9,
  dampening = 0,
  nesterov = true
}

adam_params = {
  learningRate = 0.001,
  weightDecay = 0.0005,
  beta1 = 0.9,
  beta2 = 0.999
}

rmsprop_params = {
  learningRate = 1e-2,
  weightDecay = 0.0005,
  alpha = 0.9
}

-- Log results to files
savePath = '/raid/qingnan/codes/intrinsic/'
-- trainLogger = optim.Logger(paths.concat(savePath, string.format('train_%s.log',postfix)))

local file = '/raid/qingnan/codes/qingnan_codes/train_MIT_combine.lua'
local f = io.open(file, "rb")
local line = f:read("*all")
f:close()
-- trainLogger:add{'*******************train file*******************'}
-- trainLogger:add{line}
-- trainLogger:add{'*******************train file*******************'}
print('*******************train file*******************')
print(line)
print('*******************train file*******************')

local file = '/raid/qingnan/data/MIT_BarronSplit_train.txt'
local trainSet = {}
local f = io.open(file, "rb")
while true do
  local line = f:read()
  if line == nil then break end
  table.insert(trainSet, line)
end
f:close()
local trainsetSize = #trainSet

local file = '/raid/qingnan/data/MIT_BarronSplit_fullsize_test.txt'
local testSet = {}
local f = io.open(file, "rb")
while true do
  local line = f:read()
  if line == nil then break end
  table.insert(testSet, line)
end
f:close()
local testsetSize = #testSet

local iter = 0
local epoch_judge = false
step = function(batch_size)
  local testCount = 1
  local current_loss= 0
  local current_testloss= 0
  local count = 0
  local testcount = 0
  batch_size = batch_size or 4
  local order = torch.randperm(trainsetSize)

  for t = 1,trainsetSize,batch_size do
    iter = iter + 1
    local size = math.min(t + batch_size, trainsetSize + 1) - t

    local feval = function(x_new)
      -- reset data
      if parameters ~= x_new then parameters:copy(x_new) end
      gradParameters:zero()

      local loss= 0
      for i = 1,size do
        local inputFile =  trainSet[order[t+i-1]]

        local albedoFile = string.gsub(inputFile,'input','reflectance')
        local shadingFile = string.gsub(inputFile,'input','shading')
        local tempInput = image.load(inputFile):cuda()
        local height = tempInput:size(2)
        local width = tempInput:size(3)
        local input = torch.CudaTensor(1, 3, height, width)
        local albedo = torch.CudaTensor(1, 3, height, width)
        local shading = torch.CudaTensor(1, 3, height, width)
        input[1] = image.load(inputFile):cuda()
        albedo[1] = image.load(albedoFile):cuda()
        shading[1] = image.load(shadingFile):cuda()
        input = input * 255
        albedo = albedo * 255
        shading = shading * 255
        local xGradA = albedo:narrow(4,2,width-1) - albedo:narrow(4,1,width-1)
        local yGradA = albedo:narrow(3,2,height-1) - albedo:narrow(3,1,height-1)
        local xGradS = shading:narrow(4,2,width-1) - shading:narrow(4,1,width-1)
        local yGradS = shading:narrow(3,2,height-1) - shading:narrow(3,1,height-1)
        local label = {shading,xGradS,yGradS,albedo,xGradA,yGradA,model_edge:forward(albedo),albedo}
        
        local pred = model:forward(input)
        local tempLoss =  criterion:forward(pred, label)
        local grad = criterion:backward(pred, label)
        model:backward(input, grad)

        loss = loss + tempLoss
      end
      gradParameters:div(size)
      loss= loss/size

      return loss, gradParameters
    end
    
    if epoch_judge then
      adam_params.learningRate = adam_params.learningRate*0.1
      _, fs, adam_state_save = optim.adam_state(feval, parameters, adam_params, adam_params)
      epoch_judge = false
    else
      _, fs, adam_state_save = optim.adam_state(feval, parameters, adam_params)
    end
    -- _, fs = optim.sgd(feval, parameters, sgd_params)

    count = count + 1
    current_loss= current_loss+ fs[1]
    print(string.format('Iter: %d Current loss: %4f', iter, fs[1]))
    -- trainLogger:add{string.format('Iter: %d Current loss: %4f', iter, fs[1])}

    if iter % 10 == 0 then
      local testloss= 0
      for i = 1,size do
        local inputFile = testSet[testCount]

        local albedoFile = string.gsub(inputFile,'input','reflectance')
        local shadingFile = string.gsub(inputFile,'input','shading')
        local tempInput = image.load(inputFile):cuda()
        local height = tempInput:size(2)
        local width = tempInput:size(3)
        local input = torch.CudaTensor(1, 3, height, width)
        local albedo = torch.CudaTensor(1, 3, height, width)
        local shading = torch.CudaTensor(1, 3, height, width)
        input[1] = image.load(inputFile):cuda()
        albedo[1] = image.load(albedoFile):cuda()
        shading[1] = image.load(shadingFile):cuda()
        input = input * 255
        albedo = albedo * 255
        shading = shading * 255
        local label = {shading,albedo}

        local pred = model:forward(input)
        pred = {pred[1],pred[8]}
        local tempLoss =  criterion_test:forward(pred, label)
        testloss = testloss + tempLoss

        testCount = testCount + 1
        if testCount >  testsetSize then
          testCount = 1
        end
      end
      testloss= testloss/size
      testcount = testcount + 1
      current_testloss= current_testloss+ testloss

      print(string.format('TestIter: %d Current loss: %4f', iter, testloss))
      -- trainLogger:add{string.format('TestIter: %d Current loss: %4f', iter, loss)}
    end
  end

  -- normalize loss
  return current_loss/ count, current_testloss/ testcount
end

netfiles = '/raid/qingnan/codes/intrinsic/netfiles/'
timer = torch.Timer()
local bestTestLoss = 999999
do
  for i = 1,max_iters do
    localTimer = torch.Timer()
    local train_loss,test_loss= step(batch_size,i)

    if i <= 450 then
      if bestTestLoss > test_loss then
        bestTestLoss = test_loss
        local filename = string.format('%smodel_%s_best.net',netfiles,postfix)
        model:clearState()
        torch.save(filename, model)
      end
    else
      if bestTestLoss > test_loss then
        bestTestLoss = test_loss
        local filename = string.format('%smodel_%s_best_joint.net',netfiles,postfix)
        model:clearState()
        torch.save(filename, model)
      end
    end
    
    if i == 450 then
      epoch_judge = true

      local filename = string.format('%smodel_%s_best.net',netfiles,postfix)
      model = torch.load(filename)
      model = model:cuda()
      model:training()

      parameters, gradParameters = model:getParameters()
      collectgarbage()
      
      bestTestLoss = 999999
    end

    print(string.format('Epoch: %d Current loss: %4f', i, train_loss))
    -- trainLogger:add{string.format('Epoch: %d Current loss: %4f', i, loss)}
    print(string.format('Epoch: %d Current test loss: %4f', i, test_loss))
    -- trainLogger:add{string.format('Epoch: %d Current test loss: %4f', i, testloss)}

    -- save/log current net
    local filename = string.format('%smodel_%s_%d.net',netfiles,postfix,i)
    model:clearState()
    torch.save(filename, model)
    local filename = string.format('%sstate_%s_%d.t7',netfiles,postfix,i)
    torch.save(filename, adam_state_save)
    print('Time elapsed (epoch): ' .. localTimer:time().real/(3600) .. ' hours')
    -- trainLogger:add{'Time elapsed (epoch): ' .. localTimer:time().real/(3600) .. ' hours'}
  end
end
print('Time elapsed: ' .. timer:time().real/(3600*24) .. ' days')
-- trainLogger:add{'Time elapsed: ' .. timer:time().real/(3600*24) .. ' days'}  
