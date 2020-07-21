import argparse

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument("-N",type=int,default=5)


    args = parser.parse_args()
    
    num_lines = sum(1 for line in open(args.file))
    
    line_per = int(num_lines / args.N)
    ret = num_lines % args.N
    print(line_per,ret)
    
    ret_ = 1
    _ret = 0
    for i,line in enumerate(open(args.file)):
        if ret != 0:
             if  i % (line_per + 1) == 0:
                fw = open(args.file + "." + str(int(i /(line_per))),"w")
        else:
             if  i % line_per == 0:
                fw = open(args.file + "." + str(int(i /(line_per))),"w")
        
        fw.write(line)
            

if __name__ == '__main__':
    main()
