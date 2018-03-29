local THNN = require 'nn.THNN'
local DomainTransform, parent = torch.class('nn.DomainTransform', 'nn.Module')

function DomainTransform:__init(num_iter, sigma_range, sigma_spatial, average)
  parent.__init(self)

  self.num_iter = num_iter
  self.sigma_range = sigma_range
  self.sigma_spatial = sigma_spatial
  self.average = average or 0
end

function DomainTransform:updateOutput(input)
  if self.average == 1 then
    data = input[{{},{1},{},{}}]
    edge = input[{{},{2},{},{}}]
  else
    data = input[{{},{1,3},{},{}}]
    edge = input[{{},{4},{},{}}]
  end
  
  self.output = torch.CudaTensor():resizeAs(data):copy(data)
  self.weightMap = torch.CudaTensor()
  self.inter = torch.CudaTensor()
  input.THNN.DomainTransform_updateOutput(
    edge:cdata(),
    self.output:cdata(),
    self.weightMap:cdata(),
    self.inter:cdata(),
    self.num_iter, self.sigma_range, self.sigma_spatial
  )
  -- print(self.output:size())
  --  print(torch.sum(self.forwardTarNum)/(224*224))
  return self.output
end

function DomainTransform:updateGradInput(input, gradOutput)
    if self.average == 1 then
      edge = input[{{},2,{},{}}]
      self.gradInput = torch.CudaTensor():resizeAs(input):zero()
      gradData = self.gradInput[{{},{1},{},{}}]:copy(gradOutput)
      gradEdge = self.gradInput[{{},{2},{},{}}]
    else
      edge = input[{{},{4},{},{}}]
      self.gradInput = torch.CudaTensor():resizeAs(input):zero()
      gradData = self.gradInput[{{},{1,3},{},{}}]:copy(gradOutput)
      gradEdge = self.gradInput[{{},{4},{},{}}]
    end
    
    self.gradweightMap = torch.CudaTensor():resizeAs(self.weightMap):zero()
    input.THNN.DomainTransform_updateGradInput(
      edge:cdata(),
      gradData:cdata(),
      gradEdge:cdata(),
      self.weightMap:cdata(),
      self.inter:cdata(),
      self.gradweightMap:cdata(),
      self.num_iter, self.sigma_range, self.sigma_spatial
    )
    -- print(self.gradInput:size())
    return self.gradInput
end

function DomainTransform:clearState()
  nn.utils.clear(self, 'weightMap', 'inter', 'gradweightMap')
  return parent.clearState(self)
end