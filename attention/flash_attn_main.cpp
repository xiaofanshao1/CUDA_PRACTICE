#include <torch/torch.h>
#include <iostream>

torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V);
int main(){
    const int batch_size =2;
    const int head_num=32;
    const int seq_len=10;
    const int head_embed=4096;

    auto q=torch::randn({batch_size,head_num,seq_len,head_embed}).cuda();
    auto k=torch::randn({batch_size,head_num,seq_len,head_embed}).cuda();
    auto v=torch::randn({batch_size,head_num,seq_len,head_embed}).cuda();

    forward(q,k,v);
    
    return 0;
    
}