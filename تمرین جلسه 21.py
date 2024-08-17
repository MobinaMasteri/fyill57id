class names:
    def __init__(self,n):
        self.n=n
        x=[]
        for i in range(n):
            name=input("name=")
            x.append(name)
        self.x=x
        p1=names(n)
    def max(self):
        print(max(p1.x))
    def min(self):
        print(min(p1.x))

name=input("names=")
p1=names()
p1.max()
p1.min()
        
