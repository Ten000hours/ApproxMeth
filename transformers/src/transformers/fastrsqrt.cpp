#include <iostream>
#include <torch/extension.h>
#include <bits/stdc++.h>
#include <cmath>

using namespace std;

typedef union {
  float f;
  struct {
    unsigned int mantisa : 23;
    unsigned int exponent : 8;
    unsigned int sign : 1;
  } parts;
} float_cast;
// typedef union {
//   int f;
//   struct {
//     unsigned int mantisa : 23;
//     unsigned int exponent : 8;
//   } parts;
// } int_cast;
// float sqrts(float x){
//     return sqrt(x);
// }
// float rsqrts(float x){
//     return 1/sqrt(x);
// }
torch::Tensor InvSqrt(torch::Tensor x) {
    // auto xhalf = 0.5f*x;
    // int* x_data = x.data<int>();
    // int i = *x_data;
    // i = 0x5f3759df - (i >> 1);
    // float result = *reinterpret_cast<float*>(&i);
    // return torch::from_blob(&result, 1, x.options());
    x = x.to(torch::kCPU);
    int64_t num_elements = x.numel();
    // cout<< "where is x: "<< x.device()<<endl;

    // Create an output tensor with the same shape and data type as the input tensor
    torch::Tensor output = torch::empty_like(x);


    // Access data as float pointers
    float* x_data = x.data_ptr<float>();
    // cout<<"x_data [1] : "<<x_data[1]<<endl;
    float* output_data = output.data_ptr<float>();

    for (int64_t i = 0; i < num_elements; ++i) {
        float x_value = x_data[i];

        // Apply the Fast Inverse Square Root algorithm
        int i_as_int = *reinterpret_cast<int*>(&x_value);
        i_as_int = 0x5f3759df - (i_as_int >> 1);
        float result = *reinterpret_cast<float*>(&i_as_int);

        for (int64_t j = 0; j < 1; ++j) {
          result = result * (1.5f - 0.5f * x_value * result * result);
        }


        output_data[i] = result; 
    }
    output = output.to(torch::kCUDA);

    return output;
}

// float randomFloat()
// {
//     return (float)(rand()) / (float)(RAND_MAX);
// }
 
int randomInt(int a, int b)
{
    if (a > b)
        return randomInt(b, a);
    if (a == b)
        return a;
    return a + (rand() % (b - a));
}
 
// float randomFloat(int a, int b)
// {
//     if (a > b)
//         return randomFloat(b, a);
//     if (a == b)
//         return a;
 
//     return (float)randomInt(a, b) + randomFloat();
// }

float RandomFloat(float a, float b) {
    float random = ((float) rand()) / (float) RAND_MAX;
    float diff = b - a;
    float r = random * diff;
    return a + r;
}
torch::Tensor SqrtMod(torch::Tensor input) {
    float x2 = 0.0f; 
    input = input.to(torch::kCPU);
    int64_t num_elements = input.numel();

    // Create an output tensor with the same shape and data type as the input tensor
    torch::Tensor output = torch::empty_like(input);

    // Access data as float pointers
    _Float32* input_data = input.data_ptr<_Float32>();
    // cout<<"x_data [1] : "<<x_data[1]<<endl;
    _Float32* output_data = output.data_ptr<_Float32>();
    int x1_int =0;
    float x1_frac =0.0;

    // create shares for 2PC(difference between shares within ±5 on exponent)
    for (int64_t i = 0; i < num_elements; ++i) {
        if (isnan(input_data[i]))
        {
            output_data[i] = input_data[i];
            continue;
        }
        float input_value = input_data[i];
        int input_int = (int) input_value; 
        // if(input_int == 1){

        //     x1_int = randomInt(0, input_int);
        //     x1_frac = RandomFloat(0.1, input_value-0.4);

        // }else{
        //     x1_int =  (int)(input_int/2);
        //     x1_frac = RandomFloat(0.1, input_value-input_int -0.1);

        // }
        // float x1 = x1_frac + x1_int;
        float x1 = (float)(rand()) / (float)(RAND_MAX);

        int i1 = *(int*)(&x1);
        x2 = (input_value - x1);
        int i2 = abs(*(int*)(&x2));
        i1 =  (i1 >> 2) + 0x1ffd1df6;
        i2 =  (i2 >> 2);  
        int temp = abs(i1 + i2);
        float x = *(float*)(&temp);
        float inital_approxi = x;
        for(int64_t j =0; j< 4; ++j)
        {
            // x = x*(1.5f - fmod(x1+x2, 1.9001)*0.5f * pow(x,2));
            x = 0.5f * (x + ((x1 + x2)) / x);
        }
        // if(abs(x - 1/sqrt(input_value)) > 0.1){
        //   cout << "precision > 0.1 with input= " << input_value
        //        << "|| approxi res: " << x
        //        << "|| real rsqrt: " << 1 / sqrt(input_value) << "|| x1 = " << x1
        //        << "||x2 = " << x2 << endl;
        //   throw invalid_argument("precision large than 0.1");
        //   exit(EXIT_FAILURE);
        // }
        if(isinf(x)||isnan(x)){
            cout<< "input = "<< input_value << endl;
            cout<< "x1_int = "<< x1_int << endl;
            cout<< "x1_frac = "<< x1_frac << endl;
            cout<< "x1 = "<< x1 << endl;
            cout<< "x2 = "<< x2 << endl;
            cout<< "inital approx = "<< inital_approxi << endl;
            cout<< "x = "<< x << endl;
            cout<< "==========================" << endl;
            throw invalid_argument("nan or inf result ");
            exit(EXIT_FAILURE);
        }
        output_data[i] = x;
    }
    output = output.to(torch::kCUDA);
    // x = x*(1.5f - xhalf*x*x);
    // x = (x1+x2)*(1.5f - (x1half + x2half)*(x1*x1 + x2*x2 +2*x1*x2));
    return output;
}
float findFirstValidDigit(float num) {
    // Check for special cases like zero or infinity
    if (std::isnan(num) || std::isinf(num) || num == 0.0) {
        return 0.0;
    }

    // cout<< "input = "<< num<< endl;
    // Take the absolute value to handle negative numbers
    float absoluteNum = std::abs(num);

    // Calculate the exponent of the number
    int exponent = static_cast<int>(std::floor(std::log10(absoluteNum)));
    // cout<<"exponent = "<< exponent<<endl;

    // Calculate the first valid digit
    float firstValidDigit = 5 * std::pow(10.0, exponent-1);
    // cout<<"output = "<< firstValidDigit<<endl;

    return firstValidDigit;
}
float sharetest(float input)
{
    float_cast x1 = {.f=0.0};
    x1.parts.exponent= 121;
    x1.parts.mantisa = 2348810;
    x1.parts.sign =0;
    int x1int = x1.parts.exponent*pow(2,23) + x1.parts.mantisa;
    float x1f = *(float*)(&x1int);

    float_cast inputcast = {.f=input};
    int inputint = inputcast.parts.exponent*pow(2,23) + inputcast.parts.mantisa;

    int anothershare = inputint - x1int;

    float x2 = *(float*)(&anothershare);
    cout<< "x2 = "<< x2<< "?= "<< input - x1f<<endl;
    return 0.0f;
}
torch::Tensor rSqrtMod(torch::Tensor input) {
    float x2 = 0.0f; 
    input = input.to(torch::kCPU);
    int64_t num_elements = input.numel();

    // Create an output tensor with the same shape and data type as the input tensor
    torch::Tensor output = torch::empty_like(input);

    // Access data as float pointers
    _Float32* input_data = input.data_ptr<_Float32>();
    // cout<<"x_data [1] : "<<x_data[1]<<endl;
    _Float32* output_data = output.data_ptr<_Float32>();
    int x1_int =0;
    float x1_frac =0.0;
    float x1 = 0.0;

    // create shares for 2PC(difference between shares within ±5 on exponent)
    for (int64_t i = 0; i < num_elements; ++i) {
        if (isnan(input_data[i]))
        {
            output_data[i] = input_data[i];
            continue;
        }
        float input_value = input_data[i];
        int input_int = (int) input_value; 
        for (size_t i = 0; i < 100; i++)
        {
          if (input_int == 1) {
            x1_int = randomInt(0, input_int);
            x1_frac = RandomFloat(0.1, input_value-0.4);

          } else {
            // x1_int =  (int)(input_int/2);
            x1_int =  randomInt(0, 100);
            // x1_int = 1000;
            x1_frac = RandomFloat(findFirstValidDigit(input_value-input_int), input_value - input_int -
                                           findFirstValidDigit(input_value- input_int));
            // x1_frac = 0.0f;
          }
          x1 += x1_frac + x1_int;
        }
        x1 = x1/100;
        
        
        float_cast x1c={.f=x1};
        // int i1 = *(int*)(&x1);
        int i1 = x1c.parts.exponent * pow(2, 23) + x1c.parts.mantisa;
        // cout<< "i1 = "<< i1 << endl;
        // int i1 = -((x1c.parts.mantisa>>1)+(x1c.parts.exponent>>1)*pow(2,23));
        // cout<< "i1 exponent = "<< x1c.parts.exponent << endl;
        x2 = (input_value - x1);
        float_cast x2c={.f=x2};
        // int i2 =  ((x2c.parts.mantisa >>1) )+ 0x5F775C28;
        int i2 = x2c.parts.exponent * pow(2, 23) + x2c.parts.mantisa;
        // cout<< "i2 = "<< i2 << endl;
        // int i2 = *(int*)(&x2);
        // cout<< "i2 exponent = "<< x2c.parts.exponent << endl;
        int test = i1+i2;
        // cout<< "i1 +i2 = "<< *(float*)(&test) << endl;
        i1 =  -(i1 >> 2) + 0x2F7BACF1;
        // cout<< "i1 = "<< i1 << endl;
        i2 =  -(i2 >> 2) + 0x2F7BACF2;
        // cout<< "i2 = "<< i2 << endl;
        int temp = i1 + i2 ;
        // if (input_value >1)
        // {
            // temp += 0x1A00000;
        // }else{
            // temp -=0xC;
            // temp += 0x3200000;
        // }
        
        // int mantissa = temp & 0x7FFFFF;
        // int exponent = (temp >> 23) & 0xFF;
        // int sign =0;
        // int combine = (sign <<31) | ((exponent)<<23)|mantissa;
        // float x = *reinterpret_cast<float*>(&combine);
        // cout<< "combine = "<< x << endl;


        float x = *(float*)(&temp);
        // cout<<"compare "<<x<<endl;
        float inital_approxi = x;
        for(int64_t j =0; j<6; ++j)
        {
            // x = x*(1.5f - fmod(x1+x2, 1.9001)*0.5f * pow(x,2));
            x = x*(1.5f - 0.5f *(input_value) * x * x);
        }
        // if(abs(x - 1/sqrt(input_value)) > 0.1){
        //   cout << "precision > 0.1 with input= " << input_value
        //        << "|| approxi res: " << x
        //        << "|| real rsqrt: " << 1 / sqrt(input_value) << "|| x1 = " << x1
        //        << "||x2 = " << x2 << endl;
        //   throw invalid_argument("precision large than 0.1");
        //   exit(EXIT_FAILURE);
        // }
        if(isinf(x)||isnan(x)){
            cout<< "input = "<< input_value << endl;
            cout<< "x1_int = "<< x1_int << endl;
            cout<< "x1_frac = "<< x1_frac << endl;
            cout<< "x1 = "<< x1 << endl;
            cout<< "x2 = "<< x2 << endl;
            cout<< "x1+ x2 = "<< x1+x2 << endl;
            cout<< "inital approx = "<< inital_approxi << endl;
            cout<< "x = "<< x << endl;
            cout<< "==========================" << endl;
            throw invalid_argument("nan or inf result ");
            exit(EXIT_FAILURE);
        }
        output_data[i] = abs(x);
    }
    output = output.to(torch::kCUDA);
    // x = x*(1.5f - xhalf*x*x);
    // x = (x1+x2)*(1.5f - (x1half + x2half)*(x1*x1 + x2*x2 +2*x1*x2));
    return output;
}

torch::Tensor poly2Approx(torch::Tensor x){
    x = x.to(torch::kCPU);
    int64_t num_elements = x.numel();

    // Create an output tensor with the same shape and data type as the input tensor
    torch::Tensor output = torch::empty_like(x);

    // Access data as float pointers
    float* x_data = x.data_ptr<float>();
    float* output_data = output.data_ptr<float>();

    for (int64_t i = 0; i < num_elements; ++i) {
        float x_value = x_data[i];

        float result = 0.0f;
        result = 1 - x_value / 2 + 0.5f + 3 * pow(x_value - 1, 2) / 8;

        for (int64_t j = 0; j < 1; ++j) {
          result = result * (1.5f - 0.5f * x_value * result * result);
        }


        output_data[i] = result; 
    }
    output = output.to(torch::kCUDA);

    return output;
}
torch::Tensor poly7Approx(torch::Tensor x){
    x = x.to(torch::kCPU);
    int64_t num_elements = x.numel();

    // Create an output tensor with the same shape and data type as the input tensor
    torch::Tensor output = torch::empty_like(x);

    // Access data as float pointers
    float* x_data = x.data_ptr<float>();
    float* output_data = output.data_ptr<float>();

    for (int64_t i = 0; i < num_elements; ++i) {
        float x_value = x_data[i];

        float result = 0.0f;
        result = 1 - x_value/ 2 + 0.5f + 3 * pow(x_value - 1, 2) / 8 -
                 5 * pow(x_value - 1, 3) / 16 + 35 * pow(x_value - 1, 4) / 32 -
                 63 * pow(x_value - 1, 5) / 256 + 231 * pow(x_value - 1, 6) / 1024 -
                 429 * pow(x_value - 1, 7) / 2048;

        for (int64_t j = 0; j < 3; ++j) {
          result = result * (1.5f - 0.5f * x_value * result * result);
        }


        output_data[i] = result; 
    }
    output = output.to(torch::kCUDA);

    return output;
}
torch::Tensor Fastexp(torch::Tensor input){
    //magic number = 0x3f7a3bed
    input = input.to(torch::kCPU);
    int64_t num_elements = input.numel();

    // Create an output tensor with the same shape and data type as the input tensor
    torch::Tensor output = torch::empty_like(input);

    // Access data as float pointers
    _Float32* input_data = input.data_ptr<_Float32>();
    // cout<<"x_data [1] : "<<x_data[1]<<endl;
    _Float32* output_data = output.data_ptr<_Float32>();
    for (int64_t i = 0; i < num_elements; i++)
    {
      float x1 = (float)(rand()) / (float)(RAND_MAX);
    //   int i1 = *(int*)&x1;
      float x2 = input_data[i] - x1; 
    //   int i2 = *(int*)&x2;
      int temp1 = x1 * 0xb8aa3b + 0x3f7a3bed;
      int temp2 = x2 * 0xb8aa3b;
      // i1 = *(int*)&temp1;
      // i2 = *(int*)&temp2;
      int temp = temp1 + temp2;
      float x = *(float*)&temp;

    
    // for (int64_t j = 0; j < 1; ++j) {
    //     x = x - (exp(x) - input_data[i]);
    // }

      output_data[i] = x;
    }
    output = output.to(torch::kCUDA);
    // x1 = *(float*)&temp1;
    // x2 = *(float*)&temp2;
    return output;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fastrsqrt2PC", &rSqrtMod, "2pc Fast Inverse Square Root");
    m.def("share", &sharetest, "share test");
    m.def("fastsqrt2PC", &SqrtMod, "2pc Fast Square Root");
    m.def("fastrexp2PC", &Fastexp, "2pc Fast exp");
    m.def("poly7Approx2PC", &poly7Approx, "7-degree polynomial approximation 2PC");
    m.def("poly2Approx2PC", &poly2Approx, "2-degree polynomial approximation 2PC");
    m.def("fastrsqrt", &InvSqrt, "Fast Inverse Square Root");
}
// int rsqrtNewton(float init, float input){
//     int iter = 0;
//     float x = init;
//     while (abs(rsqrts(input)- x) > 0.001)
//     {
//         x = x*(1.5f - fmod(input, 1.9001)*0.5f * pow(x,2));
//         // cout<< "x: "<< x << endl;
//         iter++;
//     }
//     return iter;
 
// }
// float SqrtMod(float x1, float x2, float input) {
//     float x1half = 0.5f*x1;
//     float x2half = 0.5f*x2;
//     // cout<< " x1= : "<< static_cast<int>(std::round(x1*10000)) << endl;
//     // cout<< " x2= : "<< static_cast<int>(std::round(x2*10000)) << endl;
//     int i1 = *(int*)&x1;
//     cout<< " i1= : "<< i1 << endl;
//     int i2 = *(int*)&x2;
//     cout<< " i2= : "<< i2 << endl;
//     i1 =  (i1 >> 2) + 0x1ffd1df6;
//     i2 =  (i2 >> 2);
//     // x = x*(1.5f - xhalf*x*x);
//     // x = (x1+x2)*(1.5f - (x1half + x2half)*(x1*x1 + x2*x2 +2*x1*x2));
//     int temp = i1 + i2;
//     float x = *(float*)&temp;
//     cout<< "inital approximation : "<< x << endl;
//     int iter = 0;
//     while (abs(sqrts(input)- x) > 0.001)
//     {
//         x = 0.5f * (x +fmod((x1 + x2), 6.5536)/ x);
//         // cout<< "x: "<< x << endl;
//         iter++;
//     }
//     cout<<"iter: "<< iter<< endl;
//     return x;
// }

// float exps(float x){
//     float res = exp(x);
//     return res;
// }

// float Fastexp(float x1, float x2){
//     //magic number = 0x3f7a3bed
//     int i1 = *(int*)&x1;
//     int i2 = *(int*)&x2;
//     int temp1 = x1*0xb8aa3b + 0x3f7a3bed;
//     int temp2 = x2*0xb8aa3b;
//     // i1 = *(int*)&temp1;
//     // i2 = *(int*)&temp2;
//     int temp = temp1 + temp2;
//     float x =  *(float*)&temp;
//     // x1 = *(float*)&temp1;
//     // x2 = *(float*)&temp2;
//     return x;
// }


// int main(int argc, char **argv) {
//     float x = 1.083;
//     // cout<< " x= : "<< static_cast<int>(std::round(x*10000)) << endl;
//     int i = *(int*)&x;
//     cout<< " i= : "<< i << endl;
//     float x2 =//0.7912;//0.6212;//0.01021;//0.7916;//0.57281; 0.7916; 
//     // 1.23E-7;
//     1.011;
//     float x1 = x-x2;//6.5016;//x - x1;
//     // cout<< "sqrt: "<< sqrts(x)<< endl;
//     // cout<< "fast sqrt: "<<SqrtMod(x1, x2,x)<< endl;
//     // cout<< "difference: "<< sqrts(x) - SqrtMod(x1, x2, x)<< endl; 
//     // cout<< "exp: "<<exps(x)<< endl;
//     // cout<< "fast exp : "<<Fastexp(x2, x1)<< endl;
//     // cout<< "difference: "<< exps(x) - Fastexp(x1,x2)<< endl; 

//     cout<< "fast rsqrt: "<<rsqrts(x)<< endl;
//     cout<< "fast rsqrt share : "<<rSqrtMod(x1, x2, x)<< endl;
//     cout<< "difference: "<< rsqrts(x) - rSqrtMod(x1, x2, x)<< endl; 


//     cout<< " poly newton iter(degree 7): "<< rsqrtNewton(2.6215,x) << endl;
//     cout<< " poly newton iter(degree 4): "<< rsqrtNewton(2.4188,x) << endl;
//     cout<< " poly newton iter(degree 2): "<< rsqrtNewton(1.007,x) << endl;
// }
