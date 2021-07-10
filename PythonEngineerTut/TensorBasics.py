import torch

def initialization_demo():
    x = torch.empty(2,3)
    y = torch.rand(2,3)
    z = torch.zeros(2,3)
    a = torch.ones(2,3, dtype=torch.int32)

    print(x.dtype)
    print(a.dtype)
    print(y.size())

def add_demo():
    x = torch.rand(2,3)
    y = torch.rand(2,3)
    z = x+y #Element wise addition
    print(z)
    y.add_(x) #In-place addition
    print(y)

def slicing_demo():
    x = torch.rand(5,4)
    print(x)
    print(x[1:3,1:3])
    print(x[1,1])
    print(x[1,1].item()) #Works only for single element

if __name__=="__main__":
    #print("Checking Initialization function")
    #initialization_demo()
    #print("Checking addition function")
    #add_demo()
    print("Checking slicing demo")
    slicing_demo()
