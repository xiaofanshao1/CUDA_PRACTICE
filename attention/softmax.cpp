#include <iostream>
#include <vector>
#include <cmath><
#include <limits>
using namespace std;

/**
 * const 放置
 * vector<float> const &src | const vector<float>  &src  作用相同
 * vector<float>* const src | const vector<float>* src 作用不同
 *      修饰src，src是常亮指针不能修改，但是可以通过src修改vector内元素
 *      修改vector，指针可以指向其他对象，但是 src 目前指向常量vector，不能修改
 * */
vector<float> nativeSoftmax(vector<float> const &src){
    vector<float> res(src.size(),0);
    float sum=0.f;
    for(int i=0;i<src.size();i++){
        //溢出范围 
        sum+=src[i];
    }

    for(int i=0;i<src.size();i++){
        res[i]=exp(src[i])/sum;
    }
    return res;
}

vector<float> safeSoftmax1(const vector<float> &src){
    vector<float> res(src.size(),0);
    //? min float
    float tmax=src[0];
    float sum=0.f;
    for(int i=0;i<src.size();i++){
        tmax=max(tmax,src[i]);

    }
    //这个地方无法和上面合并  sum+=src[i];可能超出边界
    for(int i=0;i<src.size();i++){
        sum+=exp(src[i]-tmax);
    }
    for(int i=0;i<src.size();i++){
        res[i]=exp(src[i]-tmax)/(sum);
    }
    return res;
}

// online 尝试把 sum和max一次循环
vector<float> onlineSoftmax(const vector<float> &src){
    vector<float> res(src.size(),0);
    float tmax=src[0];
    float sum=0.f;
    for(int i=0;i<src.size();i++){
        if(src[i]>tmax){
            tmax=src[i];
            sum=sum*exp(tmax-src[i])+1;
        }
        else   
            sum+=exp(src[i]-tmax)+exp(src[i]-tmax);
    }
    for(int i=0;i<src.size();i++){
        res[i]= exp(src[i]-tmax)/sum;
    }
    return res;
}

vector<float> onlineSoftmax_dot_value(const vector<float> &src,const vector<float> &value){
    vector<float> res(src.size(),0);
    float tmax=src[0];
    float sum=0.f;
    for(int i=0;i<src.size();i++){
        if(src[i]>tmax){
            tmax=src[i];
            sum=sum*exp(tmax-src[i])+1;
        }
        else   
            sum+=exp(src[i]-tmax)+exp(src[i]-tmax);
    }
    for(int i=0;i<src.size();i++){
        res[i]= exp(src[i]-tmax)/sum* value[i];// only change here,try to melt this loop next step
    return res;

} 
float onlineSoftmax_onlineDot(const vector<float> &src,const vector<float> &value){
    float sum=0.f;
    float last_sum=0.f;
    float last_tmax=numeric_limits<float>::min();
    float tmax=numeric_limits<float>::min();
    float res=0.fs;

    for(int i=0;i<src.size();i++){
        tmax=max(src[i],tmax);
        sum=sum *exp(last_tmax-tmax)+exp(src[i]-tmax);
        
        res=res*last_sum/sum*exp(last_tmax-tmax)+exp(src[i]-tmax)/sum*value[i];

        last_sum=sum;
        last_tmax=tmax; 
    }
    return res;
} 


int main(){
    vector<float> src{1.0f,3.25f,-9.33f,5.55f,1.24f};
    vector<float> value{1.0f,3.25f,-9.33f,5.55f,1.24f};

    auto a=nativeSoftmax(src);
    auto b=safeSoftmax1(src);
    auto c=onlineSoftmax(src);
    auto c=onlineSoftmax_dot_value(src,value);
    
    for(int i=0;i<src.size();i++){
        cout<<a[i]<<" "<<b[i]<<" "<<c[i]<<endl;
    }

}