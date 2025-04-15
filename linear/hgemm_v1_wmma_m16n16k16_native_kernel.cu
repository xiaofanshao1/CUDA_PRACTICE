#include <iostream>
#include <cuda_runtime.h>

#include "tester.h"
#include "common.h"


using namespace nvcuda;

int main(int argc,char * argv[]){
    Tester tester(512,2048,1024,1,10,100,false);
    std::cout<<"right"<<std::endl;
    return 0;
}